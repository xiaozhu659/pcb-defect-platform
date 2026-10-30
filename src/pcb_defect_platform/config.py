"""项目配置:路径、模型位置、数据库与 Redis 连接。"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for _directory in (MODEL_DIR, DATA_DIR, UPLOAD_DIR, OUTPUT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# 模型权重(不入库,首次运行前需放到 models/ 下或通过环境变量指定)
YOLO_WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", MODEL_DIR / "yolo_best.pt"))
SAM_WEIGHTS = os.environ.get("SAM_WEIGHTS", str(MODEL_DIR / "sam2.1_t.pt"))

# 视觉大模型
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-flash"
LLM_EFFORT = os.environ.get("LLM_EFFORT", "low")  # none / low / high
if LLM_EFFORT not in {"none", "low", "high"}:
    LLM_EFFORT = "low"

# 存储与队列
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "pcb_jobs"
STALE_JOB_MINUTES = max(1, int(os.environ.get("STALE_JOB_MINUTES", "30")))

# 类别(顺序必须与训练时一致)
CLASS_ORDER = ["Missing_hole", "Mouse_bite", "Open_circuit",
               "Short", "Spur", "Spurious_copper"]
CLASS_NAMES_ZH = {
    "Missing_hole": "缺孔",
    "Mouse_bite": "鼠咬",
    "Open_circuit": "开路",
    "Short": "短路",
    "Spur": "毛刺",
    "Spurious_copper": "多余铜",
}

CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMGSZ = int(os.environ.get("IMGSZ", "1024"))
SAM_IMGSZ = int(os.environ.get("SAM_IMGSZ", "512"))
MAX_UPLOAD_MB = max(1, int(os.environ.get("MAX_UPLOAD_MB", "20")))
