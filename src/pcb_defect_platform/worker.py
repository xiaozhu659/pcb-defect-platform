"""队列任务:在后台执行一次"双方案"分析,并把结果写回数据库。"""

import json
from datetime import datetime
from pathlib import Path

from .config import OUTPUT_DIR
from .db import Job, get_session
from .pipeline import run_both


def process_job(job_id: int) -> None:
    session = get_session()
    job = session.get(Job, job_id)
    if job is None:
        session.close()
        return
    if job.status in {"done", "failed"}:
        session.close()
        return

    job.status = "running"
    job.error = None
    session.commit()

    try:
        outcome = run_both(Path(job.image_path))
        result = outcome["result"]
        image_paths = {}
        for kind, image in outcome["images"].items():
            target = OUTPUT_DIR / f"job_{job_id}_{kind}.png"
            image.save(target)
            image_paths[kind] = target.name

        result["images"] = image_paths
        yolo_result = result.get("yolo") or {}
        llm_result = result.get("llm") or {}
        job.yolo_result = json.dumps(yolo_result, ensure_ascii=False)
        job.llm_result = json.dumps(llm_result, ensure_ascii=False)
        job.yolo_ms = int(yolo_result.get("elapsed_ms") or 0)
        job.llm_ms = int(llm_result.get("elapsed_ms") or 0)
        if yolo_result.get("error") and llm_result.get("error"):
            job.status = "failed"
            job.error = "两套检测方案均执行失败"
        else:
            job.status = "done"
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        job.finished_at = datetime.now()
        session.commit()
        session.close()
