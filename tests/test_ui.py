from io import BytesIO
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image

from pcb_defect_platform.ui import _download_image


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (20, 10), "blue").save(buffer, format="PNG")
    return buffer.getvalue()


class DownloadImageTests(TestCase):
    def test_returns_pil_image_for_gradio(self) -> None:
        response = Mock(status_code=200, content=png_bytes())
        with patch("pcb_defect_platform.ui.requests.get", return_value=response):
            image = _download_image(3, "yolo", {"Authorization": "Bearer test"})

        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (20, 10))

    def test_returns_none_for_error_response(self) -> None:
        response = Mock(status_code=404, content=b"missing")
        with patch("pcb_defect_platform.ui.requests.get", return_value=response):
            image = _download_image(3, "llm", {"Authorization": "Bearer test"})

        self.assertIsNone(image)
