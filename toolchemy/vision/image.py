import base64
from PIL import Image, ImageOps, ImageDraw, UnidentifiedImageError
from io import BytesIO

from toolchemy.utils.logger import get_logger


class UnknownImageFormatError(Exception):
    pass


class ImageProcessor:
    def __init__(self, input_image: str | Image.Image):
        self._logger = get_logger()
        self._img = None
        self._image_path = None
        if isinstance(input_image, str):
            self._image_path = input_image
        else:
            self._img = input_image.copy()
            self._img.format = input_image.format
            self._img.format_description = input_image.format_description
            self._img.filename = input_image.filename
            self._img.info = input_image.info.copy()

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
        self._img.save(img_file, self._img.format or None)

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
            if self._image_path is None:
                raise ValueError(f"image_path is empty, cannot load the image")
            try:
                self._img = Image.open(self._image_path)
            except UnidentifiedImageError as e:
                raise UnknownImageFormatError(str(e))

    def _close(self):
        if self._img:
            self._img.close()
            self._img = None

    @classmethod
    def render_annotated(cls, img, boxes: list[dict] | None, color=(0, 255, 0), width=3):
        img = ImageOps.exif_transpose(img).convert("RGB")

        draw = ImageDraw.Draw(img)
        font = None  # You can load a TTF if you want consistent sizing

        for b in (boxes or []):
            x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"][:4]]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            labels = "Unknown"
            if "labels" in b and b["labels"]:
                labels = b["labels"]
                if isinstance(labels, list):
                    labels = " / ".join(labels)
            txt = labels
            tw, th = draw.textlength(txt, font=font), 12
            draw.rectangle([x1, max(0, y1 - th - 2), x1 + int(tw) + 6, y1], fill=(0, 0, 0))
            draw.text((x1 + 3, y1 - th - 1), txt, fill=(255, 255, 255), font=font)

        return img

    @classmethod
    def show_annotated(cls, img, boxes: list[dict] | None, color=(0, 255, 0), width=3):
        img = cls.render_annotated(img, boxes, color, width)
        img.show()
