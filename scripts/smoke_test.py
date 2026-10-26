"""对已部署平台执行完整的 HTTP 冒烟测试。"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


def load_env_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(base_url: str, image_path: Path, timeout_seconds: int) -> None:
    env_values = load_env_file(PROJECT_ROOT / ".env")
    username = os.environ.get("APP_USER") or env_values.get("APP_USER")
    password = os.environ.get("APP_PASSWORD") or env_values.get("APP_PASSWORD")
    require(bool(username and password), "未找到 APP_USER / APP_PASSWORD")

    base_url = base_url.rstrip("/")
    health = requests.get(f"{base_url}/health", timeout=15)
    require(health.status_code == 200, f"健康检查失败: {health.status_code}")
    health_data = health.json()
    require(health_data.get("status") == "ok", "健康检查状态不是 ok")
    print(f"[1/7] 健康检查通过,队列={health_data.get('queue')}")

    ui = requests.get(f"{base_url}/ui/", timeout=15)
    require(ui.status_code == 200, f"网页访问失败: {ui.status_code}")
    print("[2/7] Gradio 页面访问通过")

    unauthorized = requests.post(f"{base_url}/api/jobs", timeout=15)
    require(unauthorized.status_code == 401, "未登录请求没有返回 401")
    print("[3/7] JWT 未授权拦截通过")

    login = requests.post(
        f"{base_url}/api/login",
        json={"username": username, "password": password},
        timeout=20,
    )
    require(login.status_code == 200, f"登录失败: {login.status_code}")
    token = login.json().get("access_token")
    require(bool(token), "登录响应没有 token")
    headers = {"Authorization": f"Bearer {token}"}

    invalid = requests.post(
        f"{base_url}/api/jobs",
        files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")},
        headers=headers,
        timeout=20,
    )
    require(invalid.status_code == 400, "损坏图片没有返回 400")
    print("[4/7] 损坏图片被上传校验拒绝")

    suffix = image_path.suffix.lower() or ".jpg"
    content_type = "image/png" if suffix == ".png" else "image/jpeg"
    with image_path.open("rb") as image_file:
        submitted = requests.post(
            f"{base_url}/api/jobs",
            files={"file": (image_path.name, image_file, content_type)},
            headers=headers,
            timeout=60,
        )
    require(submitted.status_code == 200, f"任务提交失败: {submitted.status_code}")
    job_id = submitted.json().get("id")
    require(bool(job_id), "任务响应没有 id")
    print(f"[5/7] 登录和上传通过,任务 #{job_id}")

    deadline = time.monotonic() + timeout_seconds
    job = {}
    while time.monotonic() < deadline:
        result = requests.get(
            f"{base_url}/api/jobs/{job_id}",
            headers=headers,
            timeout=20,
        )
        require(result.status_code == 200, f"任务查询失败: {result.status_code}")
        job = result.json()
        if job.get("status") in {"done", "failed"}:
            break
        time.sleep(2)
    require(job.get("status") == "done", f"任务未成功完成: {job.get('status')}")
    print(f"[6/7] Redis 队列任务已完成: {job.get('status')}")

    for kind in ("yolo", "llm"):
        image = requests.get(
            f"{base_url}/api/files/{job_id}/{kind}",
            headers=headers,
            timeout=30,
        )
        require(image.status_code == 200, f"{kind} 结果图失败: {image.status_code}")
        require(bool(image.content), f"{kind} 结果图为空")
    print("[7/7] 两张结果图下载通过")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCB 平台端到端冒烟测试")
    parser.add_argument("--base-url", default=os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1"))
    parser.add_argument("--image", required=True, type=Path, help="用于测试的 PCB 图片")
    parser.add_argument("--timeout", type=int, default=300, help="等待任务完成的最长秒数")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.image.is_file():
        print(f"测试图片不存在: {arguments.image}", file=sys.stderr)
        raise SystemExit(2)
    try:
        run(arguments.base_url, arguments.image, arguments.timeout)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"冒烟测试失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
