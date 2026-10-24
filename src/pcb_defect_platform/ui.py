"""Gradio 页面:上传一张 PCB 图片,并排展示两种方案的检测结果。"""

import io
import time

import gradio as gr
import requests

API_BASE = "http://127.0.0.1:8000"


def login(username: str, password: str):
    """用账号密码调用 /api/login,拿到 JWT token。"""
    try:
        response = requests.post(f"{API_BASE}/api/login",
                                 json={"username": username, "password": password},
                                 timeout=15)
    except requests.RequestException as exc:
        return "", f"登录失败:{exc}"
    if response.status_code == 200:
        return response.json()["access_token"], "登录成功,可以上传图片了"
    return "", f"登录失败:{response.text[:120]}"


def analyze(image_path: str | None, token: str):
    if not image_path:
        return None, None, "请先上传一张 PCB 图片。"
    if not token:
        return None, None, "请先在上方登录。"
    headers = {"Authorization": f"Bearer {token}"}

    with open(image_path, "rb") as fh:
        response = requests.post(
            f"{API_BASE}/api/jobs",
            files={"file": ("upload.png", fh, "image/png")},
            headers=headers,
            timeout=60,
        )
    if response.status_code == 401:
        return None, None, "登录已过期,请重新登录。"
    if response.status_code != 200:
        return None, None, f"提交失败:{response.status_code} {response.text[:200]}"

    job_id = response.json()["id"]
    deadline = time.time() + 300
    job = {}
    while time.time() < deadline:
        job = requests.get(f"{API_BASE}/api/jobs/{job_id}",
                           headers=headers, timeout=15).json()
        if job["status"] in {"done", "failed"}:
            break
        time.sleep(2)

    if job.get("status") == "failed":
        return None, None, f"任务失败:{job.get('error')}"
    if job.get("status") != "done":
        return None, None, "任务超时(超过 5 分钟)"

    yolo = job.get("yolo") or {}
    llm = job.get("llm") or {}
    yolo_image = requests.get(f"{API_BASE}/api/files/{job_id}/yolo",
                              headers=headers, timeout=15).content
    llm_image = requests.get(f"{API_BASE}/api/files/{job_id}/llm",
                             headers=headers, timeout=15).content

    yolo_types = "、".join(sorted({d.get("name_zh", d.get("type", "?")) for d in yolo.get("defects", [])})) or "无"
    llm_types = "、".join(sorted({d.get("name_zh", d.get("type", "?")) for d in llm.get("defects", [])})) or "无"
    summary = (
        f"任务 #{job_id}\n"
        f"传统方案(YOLO+SAM):{yolo.get('count', 0)} 处 · {yolo_types} · {yolo.get('elapsed_ms', 0):.0f} ms\n"
        f"大模型方案(DeepSeek):{llm.get('count', 0)} 处 · {llm_types} · {llm.get('elapsed_ms', 0):.0f} ms\n"
    )
    if llm.get("summary"):
        summary += f"\n大模型总结:{llm['summary']}"
    if yolo.get("error"):
        summary += f"\n传统方案错误:{yolo['error']}"
    if llm.get("error"):
        summary += f"\n大模型错误:{llm['error']}"

    return yolo_image, llm_image, summary


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PCB 缺陷检测平台") as demo:
        gr.Markdown(
            "# PCB 缺陷检测平台\n"
            "上传一张 PCB 图片,系统会用**传统检测模型(YOLO + SAM)**和"
            "**视觉大模型(DeepSeek)**分别分析,并排展示结果。"
        )
        token_state = gr.State("")
        with gr.Row():
            user_box = gr.Textbox(label="用户名", value="admin", scale=1)
            pass_box = gr.Textbox(label="密码", type="password", scale=1)
            login_button = gr.Button("登录", scale=0)
        login_status = gr.Textbox(label="登录状态", value="未登录",
                                  interactive=False)
        with gr.Row():
            image_input = gr.Image(type="filepath", label="上传图片")
            summary_output = gr.Textbox(label="分析结果", lines=8)
        run_button = gr.Button("开始检测", variant="primary")
        with gr.Row():
            yolo_output = gr.Image(label="传统方案:YOLO 检测 + SAM 分割")
            llm_output = gr.Image(label="大模型方案:DeepSeek 判断位置")

        login_button.click(login, inputs=[user_box, pass_box],
                           outputs=[token_state, login_status])
        run_button.click(analyze, inputs=[image_input, token_state],
                         outputs=[yolo_output, llm_output, summary_output])
    return demo
