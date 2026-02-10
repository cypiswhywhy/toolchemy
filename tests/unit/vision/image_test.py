from toolchemy.utils.locations import Locations
from toolchemy.vision.image import ImageProcessor


TEST_IMAGE_PATH = Locations().in_resources("tests/vision/img1.jpg")
EXPECTED_METADATA = {
    "width": 1000,
    "height": 666,
    "size_bytes": 110151,
    "format": "JPEG",
    "format_description": "JPEG (ISO 10918)",
    "filename": TEST_IMAGE_PATH,
}


def test_image_from_path_metadata():
    with ImageProcessor(TEST_IMAGE_PATH) as image_processor:
        assert image_processor.metadata() == EXPECTED_METADATA


def test_image_from_image_metadata():
    with ImageProcessor(TEST_IMAGE_PATH) as image_processor:
        with ImageProcessor(image_processor.img) as image_processor2:
            assert image_processor2.metadata() == EXPECTED_METADATA
