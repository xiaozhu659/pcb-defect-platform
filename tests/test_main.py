from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from pcb_defect_platform.main import create_job


class FakeSession:
    def __init__(self) -> None:
        self.job = None

    def add(self, job) -> None:
        job.id = 42
        self.job = job

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


class FailingQueue:
    def enqueue(self, *args, **kwargs):
        raise ConnectionError("queue unavailable")


class FakeBackgroundTasks:
    def add_task(self, *args, **kwargs) -> None:
        raise AssertionError("should not use inline fallback when Redis enqueue fails")


def png_upload() -> UploadFile:
    buffer = BytesIO()
    Image.new("RGB", (20, 10), "green").save(buffer, format="PNG")
    buffer.seek(0)
    return UploadFile(file=buffer, filename="sample.png")


class QueueFailureTests(TestCase):
    def test_marks_job_failed_and_returns_503(self) -> None:
        session = FakeSession()
        with (
            TemporaryDirectory() as directory,
            patch("pcb_defect_platform.main.UPLOAD_DIR", Path(directory)),
            patch("pcb_defect_platform.main.get_session", return_value=session),
            patch("pcb_defect_platform.main._redis_queue", return_value=FailingQueue()),
            patch("pcb_defect_platform.main.mark_job_failed") as mark_failed,
            self.assertRaises(HTTPException) as raised,
        ):
            create_job(FakeBackgroundTasks(), png_upload(), user="admin")

        self.assertEqual(raised.exception.status_code, 503)
        mark_failed.assert_called_once()
        self.assertEqual(mark_failed.call_args.args[0], 42)
        self.assertIn("queue unavailable", mark_failed.call_args.args[1])
