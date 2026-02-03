import base64
from PIL import Image, UnidentifiedImageError
from io import BytesIO

from toolchemy.utils.logger import get_logger


class UnknownImageFormatError(Exception):
    pass


class ImageProcessor:
    def __init__(self, image_path: str):
        self._logger = get_logger()
        self._img = None
        self._image_path = image_path

    @property
    def img(self) -> Image.Image:
        self._open()
        return self._img

    @property
    def base64(self) -> str:
        self._open()
        buffered = BytesIO()
        self._img.save(buffered, format=self._img.format)  # Save as JPEG in memory
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    def metadata(self) -> dict:
        self._open()

        img_file = BytesIO()
        self._img.save(img_file, self._img.format)

        metadata = {
            "width": self._img.width,
            "height": self._img.height,
            "size_bytes": img_file.tell(),
            "format": self._img.format,
            "format_description": self._img.format_description,
            "filename": self._img.filename,
        }

        return metadata

    def scale(self, max_edge_len: int, upscale: bool = False):
        cur_w = self._img.width
        cur_h = self._img.height

        self._logger.info(f"> size (w, h): ({cur_w}, {cur_h})")

        if cur_h > cur_w:
            resize_ratio = cur_h / max_edge_len
        else:
            resize_ratio = cur_w / max_edge_len

        self._logger.info(f"> resize ratio: {resize_ratio}")

        if resize_ratio <= 1.0 and not upscale:
            self._logger.info(f"upscaling disabled, skipping")
            return

        new_h = int(cur_h // resize_ratio)
        new_w = int(cur_w // resize_ratio)

        self._logger.info(f"> new size (w, h): ({new_w}, {new_h})")

        self._img = self._img.resize((new_w, new_h))

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()

    def _open(self):
        if self._img is None:
            try:
                self._img = Image.open(self._image_path)
            except UnidentifiedImageError as e:
                raise UnknownImageFormatError(str(e))

    def _close(self):
        if self._img:
            self._img.close()
            self._img = None
