"""上传文件校验:限制大小,并确认文件确实是一张可读取的图片。"""

from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
MAX_IMAGE_PIXELS = 40_000_000
CHUNK_SIZE = 1024 * 1024


class InvalidImageError(ValueError):
    """上传内容不是受支持的图片,或图片大小超过限制。"""


def save_validated_image(
    source: BinaryIO,
    filename: str,
    target: Path,
    max_bytes: int,
) -> None:
    """分块保存上传内容,验证失败时删除临时文件。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise InvalidImageError("只支持 jpg / jpeg / png / bmp / webp 图片")

    total = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as output:
            while chunk := source.read(CHUNK_SIZE):
                total += len(chunk)
                if total > max_bytes:
                    raise InvalidImageError(
                        f"图片不能超过 {max_bytes // (1024 * 1024)} MiB"
                    )
                output.write(chunk)

        if total == 0:
            raise InvalidImageError("上传的图片是空文件")

        with Image.open(target) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise InvalidImageError("图片格式与扩展名不匹配")
            if width <= 0 or height <= 0:
                raise InvalidImageError("图片尺寸无效")
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError("图片像素总量过大")
            image.verify()
    except InvalidImageError:
        target.unlink(missing_ok=True)
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        target.unlink(missing_ok=True)
        raise InvalidImageError("文件内容不是有效图片") from exc
