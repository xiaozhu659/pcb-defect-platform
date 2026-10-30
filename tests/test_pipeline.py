from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from pcb_defect_platform.pipeline import _parse_llm_payload, detect


class Tokens(list):
    def tolist(self):
        return list(self)


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = Tokens([[1, 1, 10, 10]])
        self.cls = [0]
        self.conf = Tokens([0.9])


class FakeResult:
    def __init__(self) -> None:
        self.boxes = FakeBoxes()


class FakeYolo:
    def predict(self, *args, **kwargs):
        return [FakeResult()]


class FailingSam:
    def predict(self, *args, **kwargs):
        raise RuntimeError("segmentation failed")


class LlmPayloadTests(TestCase):
    def test_accepts_confirmed_empty_defect_list(self) -> None:
        payload = _parse_llm_payload('{"defects": [], "summary": "未发现缺陷"}')
        self.assertEqual(payload["defects"], [])

    def test_rejects_unparseable_content(self) -> None:
        with self.assertRaisesRegex(TypeError, "无法解析"):
            _parse_llm_payload("模型无法回答")

    def test_rejects_missing_defects_array(self) -> None:
        with self.assertRaisesRegex(TypeError, "缺少 defects"):
            _parse_llm_payload('{"summary": "没有按格式返回"}')

    def test_rejects_defect_without_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "没有有效对象"):
            _parse_llm_payload('{"defects": [{}], "summary": "格式错误"}')

    def test_drops_invalid_items_when_at_least_one_is_valid(self) -> None:
        payload = _parse_llm_payload(
            '{"defects": [{}, {"type": "short"}], "summary": "发现短路"}'
        )
        self.assertEqual(payload["defects"], [{"type": "short"}])


class TraditionalTimingTests(TestCase):
    def test_sam_failure_is_reported_and_timed(self) -> None:
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (32, 24), "green").save(image_path)
            with patch(
                "pcb_defect_platform.pipeline._load_models",
                return_value=(FakeYolo(), FailingSam(), None),
            ):
                result, _ = detect(image_path)

        self.assertIn("SAM分割失败", result["method"])
        self.assertEqual(result["segmentation"], "failed")
        self.assertGreaterEqual(result["sam_ms"], 0)
        self.assertGreaterEqual(result["elapsed_ms"], result["yolo_ms"])
