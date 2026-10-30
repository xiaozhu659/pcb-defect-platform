"""FastAPI 后端:上传图片、创建任务、查询结果、返回可视化图。

启动方式:
    uv run uvicorn pcb_defect_platform.main:app --host 127.0.0.1 --port 8000

任务队列:优先使用 Redis + RQ;如果本机没有 Redis,自动退化为后台线程执行,
这样可以先在 Windows 上把功能跑通,部署到服务器后再用真正的队列。
"""

import uuid
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .auth import authenticate, create_token, current_user
from .config import (
    JOB_TIMEOUT_SECONDS,
    MAX_UPLOAD_MB,
    OUTPUT_DIR,
    QUEUE_NAME,
    REDIS_URL,
    STALE_JOB_MINUTES,
    UPLOAD_DIR,
)
from .db import Job, fail_stale_jobs, get_session, init_db, mark_job_failed
from .validation import InvalidImageError, save_validated_image
from .worker import process_job

app = FastAPI(title="PCB 缺陷检测平台", version="0.1.0")


class LoginRequest(BaseModel):
    username: str
    password: str


@app.on_event("startup")
def _startup() -> None:
    init_db()
    fail_stale_jobs(STALE_JOB_MINUTES)


def _redis_queue():
    """尝试连接 Redis;失败返回 None(交给调用方走本地后台任务)。"""
    try:
        from redis import Redis
        from rq import Queue

        connection = Redis.from_url(REDIS_URL, socket_connect_timeout=1,
                                    socket_timeout=1)
        connection.ping()
        return Queue(QUEUE_NAME, connection=connection)
    except Exception:  # noqa: BLE001
        return None


@app.get("/health")
def health() -> dict:
    queue = _redis_queue()
    return {"status": "ok", "queue": "redis" if queue else "inline"}


@app.post("/api/login")
def login(payload: LoginRequest) -> dict:
    """用账号密码换取 JWT token。"""
    if not authenticate(payload.username, payload.password):
        raise HTTPException(401, "用户名或密码错误")
    return {"access_token": create_token(payload.username),
            "token_type": "bearer"}


@app.post("/api/jobs")
def create_job(background: BackgroundTasks, file: UploadFile = File(...),  # noqa: B008
               user: str = Depends(current_user)) -> dict:
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        save_validated_image(
            file.file,
            file.filename or target.name,
            target,
            MAX_UPLOAD_MB * 1024 * 1024,
        )
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from exc

    session = get_session()
    job = Job(filename=file.filename or target.name, image_path=str(target), status="queued")
    session.add(job)
    session.commit()
    job_id = job.id
    session.close()

    queue = _redis_queue()
    try:
        if queue is not None:
            queue.enqueue(process_job, job_id, job_timeout=JOB_TIMEOUT_SECONDS)
        else:
            background.add_task(process_job, job_id)
    except Exception as exc:
        message = f"任务提交队列失败:{type(exc).__name__}: {exc}"
        mark_job_failed(job_id, message)
        raise HTTPException(503, "任务队列暂时不可用,请稍后重试") from exc

    return {"id": job_id, "status": "queued",
            "queue": "redis" if queue else "inline"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int, user: str = Depends(current_user)) -> dict:
    fail_stale_jobs(STALE_JOB_MINUTES)
    session = get_session()
    job = session.get(Job, job_id)
    session.close()
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    user: str = Depends(current_user),
) -> list[dict]:
    fail_stale_jobs(STALE_JOB_MINUTES)
    session = get_session()
    jobs = (session.query(Job).order_by(Job.id.desc()).limit(limit).all())
    session.close()
    return [job.to_dict() for job in jobs]


@app.get("/api/files/{job_id}/{kind}")
def get_image(job_id: int, kind: str, user: str = Depends(current_user)) -> FileResponse:
    if kind not in {"yolo", "llm"}:
        raise HTTPException(404, "未知类型")
    path = OUTPUT_DIR / f"job_{job_id}_{kind}.png"
    if not path.exists():
        raise HTTPException(404, "结果图尚未生成")
    return FileResponse(path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<html><head><meta charset="utf-8"><title>PCB 缺陷检测平台</title></head>
<body style="font-family:sans-serif;padding:40px">
<h1>PCB 缺陷检测平台</h1>
<p>可视化页面:<a href="/ui">/ui</a></p>
<p>接口文档:<a href="/docs">/docs</a></p>
</body></html>"""


def _mount_ui():
    """把 Gradio 页面挂到 /ui,与 FastAPI 共用同一个进程和端口。"""
    import gradio as gr

    from .ui import build_ui

    return gr.mount_gradio_app(app, build_ui(), path="/ui")


app = _mount_ui()


def main() -> None:
    import uvicorn

    uvicorn.run("pcb_defect_platform.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
