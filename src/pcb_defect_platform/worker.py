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

    job.status = "running"
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
        job.yolo_result = json.dumps(result.get("yolo"), ensure_ascii=False)
        job.llm_result = json.dumps(result.get("llm"), ensure_ascii=False)
        job.yolo_ms = int((result.get("yolo") or {}).get("elapsed_ms") or 0)
        job.llm_ms = int((result.get("llm") or {}).get("elapsed_ms") or 0)
        job.status = "done"
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        job.finished_at = datetime.now()
        session.commit()
        session.close()
