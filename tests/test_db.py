from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from pcb_defect_platform.db import Base, Job, fail_stale_jobs, mark_job_failed


class DatabaseRecoveryTests(TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def test_marks_stale_active_jobs_failed(self) -> None:
        old = Job(
            filename="old.jpg",
            image_path="old.jpg",
            status="queued",
            created_at=datetime.now() - timedelta(minutes=31),
        )
        fresh = Job(
            filename="fresh.jpg",
            image_path="fresh.jpg",
            status="running",
            created_at=datetime.now(),
        )
        session = self.Session()
        session.add_all([old, fresh])
        session.commit()
        old_id = old.id
        fresh_id = fresh.id
        session.close()

        with patch("pcb_defect_platform.db.get_session", side_effect=self.Session):
            count = fail_stale_jobs(30)

        check = self.Session()
        old = check.get(Job, old_id)
        fresh = check.get(Job, fresh_id)
        check.close()
        self.assertEqual(count, 1)
        self.assertEqual(old.status, "failed")
        self.assertIsNotNone(old.finished_at)
        self.assertEqual(fresh.status, "running")

    def test_marks_queue_submission_failure(self) -> None:
        job = Job(filename="job.jpg", image_path="job.jpg", status="queued")
        session = self.Session()
        session.add(job)
        session.commit()
        job_id = job.id
        session.close()

        with patch("pcb_defect_platform.db.get_session", side_effect=self.Session):
            mark_job_failed(job_id, "queue unavailable")

        check = self.Session()
        job = check.get(Job, job_id)
        check.close()
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "queue unavailable")
