from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from pcb_defect_platform.validation import InvalidImageError, save_validated_image


def image_bytes(image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), "green").save(buffer, format=image_format)
    return buffer.getvalue()


class UploadValidationTests(TestCase):
    def test_accepts_a_real_png(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "sample.png"
            save_validated_image(BytesIO(image_bytes()), "sample.png", target, 1024 * 1024)
            self.assertTrue(target.exists())

    def test_rejects_corrupted_image(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "broken.jpg"
            with self.assertRaises(InvalidImageError):
                save_validated_image(BytesIO(b"not-an-image"), "broken.jpg", target, 1024)
            self.assertFalse(target.exists())

    def test_rejects_disallowed_extension(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "sample.txt"
            with self.assertRaises(InvalidImageError):
                save_validated_image(BytesIO(image_bytes()), "sample.txt", target, 1024)
            self.assertFalse(target.exists())

    def test_rejects_file_over_limit(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "sample.png"
            with self.assertRaises(InvalidImageError):
                save_validated_image(BytesIO(image_bytes()), "sample.png", target, 8)
            self.assertFalse(target.exists())
