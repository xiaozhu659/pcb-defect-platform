import json
from unittest import TestCase
from unittest.mock import patch

from pcb_defect_platform.db import Job
from pcb_defect_platform.worker import process_job


class FakeSession:
    def __init__(self, job: Job):
        self.job = job

    def get(self, model, job_id):
        return self.job

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


class WorkerStatusTests(TestCase):
    def test_does_not_rerun_completed_job(self) -> None:
        job = Job(filename="sample.jpg", image_path="sample.jpg", status="done")
        session = FakeSession(job)
        with (
            patch("pcb_defect_platform.worker.get_session", return_value=session),
            patch("pcb_defect_platform.worker.run_both") as run_both,
        ):
            process_job(1)

        run_both.assert_not_called()

    def test_marks_job_failed_when_both_methods_fail(self) -> None:
        job = Job(filename="sample.jpg", image_path="sample.jpg", status="queued")
        outcome = {
            "result": {
                "yolo": {"error": "model unavailable"},
                "llm": {"error": "network unavailable"},
            },
            "images": {},
        }
        session = FakeSession(job)
        with (
            patch("pcb_defect_platform.worker.get_session", return_value=session),
            patch("pcb_defect_platform.worker.run_both", return_value=outcome),
        ):
            process_job(1)

        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "两套检测方案均执行失败")
        self.assertEqual(json.loads(job.yolo_result)["error"], "model unavailable")

    def test_marks_job_done_when_one_method_succeeds(self) -> None:
        job = Job(filename="sample.jpg", image_path="sample.jpg", status="queued")
        outcome = {
            "result": {
                "yolo": {"count": 1, "defects": [], "elapsed_ms": 12},
                "llm": {"error": "network unavailable"},
            },
            "images": {},
        }
        session = FakeSession(job)
        with (
            patch("pcb_defect_platform.worker.get_session", return_value=session),
            patch("pcb_defect_platform.worker.run_both", return_value=outcome),
        ):
            process_job(1)

        self.assertEqual(job.status, "done")
        self.assertIsNone(job.error)
