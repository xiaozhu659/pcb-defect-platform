"""双方案推理管线:复用第二阶段训练好的 YOLO/SAM,并调用 DeepSeek 视觉模型。

这一层是第三阶段的核心:把第二阶段"跑脚本"的流程,变成可以被 Web 接口调用的函数。
"""

import base64
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from .config import (
    CLASS_NAMES_ZH,
    CLASS_ORDER,
    CONF_THRESHOLD,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_URL,
    IMGSZ,
    LLM_EFFORT,
    SAM_IMGSZ,
    SAM_WEIGHTS,
    YOLO_WEIGHTS,
)

PROMPT = """这是一张印刷电路板(PCB)的照片。请检查图中是否存在下列 6 类缺陷:

1. missing_hole:缺孔
2. mouse_bite:鼠咬
3. open_circuit:开路
4. short:短路
5. spur:毛刺
6. spurious_copper:多余铜

请逐个列出发现的缺陷,并给出所在区域与归一化中心坐标 center(左上角 (0,0),右下角 (1,1)),
严重程度只能是"轻微"或"明显"。只输出 JSON:
{"defects": [{"type": "missing_hole", "location": "左上", "center": [0.5, 0.5], "severity": "明显"}], "summary": "一句话总结"}"""

_yolo = None
_sam = None


def _load_models():
    """延迟加载:第一次请求时才把模型读进显存,避免启动过慢。"""
    global _yolo, _sam
    if _yolo is None:
        from ultralytics import SAM, YOLO

        if not Path(YOLO_WEIGHTS).exists():
            raise FileNotFoundError(f"缺少 YOLO 权重:{YOLO_WEIGHTS}")
        _yolo = YOLO(str(YOLO_WEIGHTS))
        try:
            _sam = SAM(SAM_WEIGHTS)
        except Exception:  # noqa: BLE001  SAM 权重缺失时退化为只画检测框
            _sam = None
    return _yolo, _sam


def _font(size: int = 18):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _downscale(image: Image.Image, max_side: int = 1600) -> Image.Image:
    """结果图给网页展示用,等比缩小到长边 1600,避免单张 7 MB 拖慢页面。"""
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, int(image.width * scale)),
                         max(1, int(image.height * scale))), Image.LANCZOS)


def detect(image_path: Path) -> tuple[dict, Image.Image]:
    """传统方案:YOLO 检测 + SAM 分割,返回结构化结果与可视化图。"""
    yolo, sam = _load_models()
    started = time.time()
    result = yolo.predict(str(image_path), imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)[0]
    elapsed = (time.time() - started) * 1000

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    boxes = result.boxes.xyxy.tolist()
    classes = [CLASS_ORDER[int(c)] for c in result.boxes.cls]
    confidences = result.boxes.conf.tolist()

    masks = []
    if sam is not None and boxes:
        try:
            # 2 GiB CPU 服务器使用 512 输入,内存在默认 1024 分辨率上容易耗尽。
            segmentation = sam.predict(
                str(image_path),
                bboxes=boxes,
                imgsz=SAM_IMGSZ,
                verbose=False,
            )[0]
            if segmentation.masks is not None:
                masks = segmentation.masks.data.cpu().numpy()
        except Exception:  # noqa: BLE001
            masks = []

    defects = []
    overlay = np.array(image).copy()
    for index, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        x1, y1, x2, y2 = box
        item = {
            "type": cls,
            "name_zh": CLASS_NAMES_ZH.get(cls, cls),
            "conf": round(float(conf), 4),
            "center": [round((x1 + x2) / 2 / width, 4), round((y1 + y2) / 2 / height, 4)],
            "box": [round(v, 1) for v in box],
        }
        if index < len(masks):
            mask = masks[index] > 0.5
            item["area_ratio"] = round(float(mask.mean()) * 100, 4)
            overlay[mask] = (overlay[mask] * 0.45 + np.array([255, 60, 60]) * 0.55).astype(np.uint8)
        defects.append(item)

    annotated = Image.fromarray(overlay)
    draw = ImageDraw.Draw(annotated)
    font = _font(max(14, width // 60))
    for defect in defects:
        x1, y1, x2, y2 = defect["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(30, 120, 220), width=3)
        label = f"{defect['name_zh']} {defect['conf'] * 100:.0f}%"
        draw.text((x1 + 2, max(0, y1 - 20)), label, fill=(30, 120, 220), font=font)

    annotated = _downscale(annotated)
    payload = {
        "method": "YOLO + SAM",
        "count": len(defects),
        "defects": defects,
        "elapsed_ms": round(elapsed, 1),
    }
    return payload, annotated


def analyze_with_llm(image_path: Path) -> tuple[dict, Image.Image]:
    """大模型方案:调用 DeepSeek 视觉模型,返回结构化结果与位置示意。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未设置环境变量 DEEPSEEK_API_KEY")

    image = Image.open(image_path).convert("RGB")
    if max(image.size) > 1536:
        scale = 1536 / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode()

    started = time.time()
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "thinking": {"type": "disabled" if LLM_EFFORT == "none" else "enabled"},
    }
    if LLM_EFFORT != "none":
        payload["reasoning_effort"] = LLM_EFFORT
    response = requests.post(
        DEEPSEEK_URL,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json=payload, timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    elapsed = (time.time() - started) * 1000
    content = data["choices"][0]["message"]["content"]
    parsed = _parse_json(content)
    defects = [d for d in parsed.get("defects", []) if isinstance(d, dict)]
    for defect in defects:
        raw = str(defect.get("type", "")).strip().lower().replace(" ", "_")
        defect["type"] = raw
        defect["name_zh"] = CLASS_NAMES_ZH.get(raw.capitalize(), raw)

    # 把大模型报出的位置画在原图上
    annotated = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = _font(max(14, annotated.width // 60))
    for defect in defects:
        center = defect.get("center")
        if not (isinstance(center, (list, tuple)) and len(center) == 2):
            continue
        try:
            cx, cy = float(center[0]) * annotated.width, float(center[1]) * annotated.height
        except (TypeError, ValueError):
            continue
        radius = max(8, annotated.width // 90)
        draw.line([cx - radius, cy - radius, cx + radius, cy + radius], fill=(200, 30, 30), width=3)
        draw.line([cx - radius, cy + radius, cx + radius, cy - radius], fill=(200, 30, 30), width=3)
        draw.text((cx + radius + 3, cy - radius - 16), defect["name_zh"],
                  fill=(200, 30, 30), font=font)

    annotated = _downscale(annotated)
    result = {
        "method": f"DeepSeek {DEEPSEEK_MODEL}(思考:{LLM_EFFORT})",
        "count": len(defects),
        "defects": defects,
        "summary": parsed.get("summary", ""),
        "elapsed_ms": round(elapsed, 1),
        "usage": data.get("usage", {}),
    }
    return result, annotated


def _parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def run_both(image_path: Path) -> dict:
    """两种方案依次执行,任一失败不影响另一个的结果。"""
    result = {"image": image_path.name}
    images = {}

    try:
        yolo_result, yolo_image = detect(image_path)
        result["yolo"] = yolo_result
        images["yolo"] = yolo_image
    except Exception as exc:  # noqa: BLE001
        result["yolo"] = {"error": f"{type(exc).__name__}: {exc}"}

    try:
        llm_result, llm_image = analyze_with_llm(image_path)
        result["llm"] = llm_result
        images["llm"] = llm_image
    except Exception as exc:  # noqa: BLE001
        result["llm"] = {"error": f"{type(exc).__name__}: {exc}"}

    return {"result": result, "images": images}
