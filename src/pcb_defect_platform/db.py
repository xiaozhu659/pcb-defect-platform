"""SQLAlchemy 数据层:每次检测请求都记录一条任务。"""

import json
from datetime import datetime, timedelta

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Job(Base):
    """一次"上传图片 → 两种方案分析"的任务记录。"""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued/running/done/failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    yolo_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    yolo_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "finished_at": self.finished_at.isoformat(timespec="seconds") if self.finished_at else None,
            "yolo": json.loads(self.yolo_result) if self.yolo_result else None,
            "llm": json.loads(self.llm_result) if self.llm_result else None,
            "yolo_ms": self.yolo_ms,
            "llm_ms": self.llm_ms,
            "error": self.error,
        }


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}
                       if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()


def mark_job_failed(job_id: int, error: str) -> None:
    """把无法提交到队列的任务明确标成失败,避免永久停在 queued。"""
    session = get_session()
    try:
        job = session.get(Job, job_id)
        if job is not None and job.status in {"queued", "running"}:
            job.status = "failed"
            job.error = error[:500]
            job.finished_at = datetime.now()
            session.commit()
    finally:
        session.close()


def fail_stale_jobs(max_age_minutes: int) -> int:
    """将长时间未结束的任务标记失败,恢复异常中断后的状态。"""
    cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
    session = get_session()
    try:
        jobs = (
            session.query(Job)
            .filter(Job.status.in_(("queued", "running")))
            .filter(Job.created_at < cutoff)
            .all()
        )
        for job in jobs:
            job.status = "failed"
            job.error = (
                f"任务超过 {max_age_minutes} 分钟仍未完成,"
                "可能由队列或 Worker 中断导致"
            )
            job.finished_at = datetime.now()
        session.commit()
        return len(jobs)
    finally:
        session.close()
