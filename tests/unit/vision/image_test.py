from toolchemy.utils.locations import Locations
from toolchemy.vision.image import ImageProcessor


def test_image_metadata():
    locations = Locations()
    test_img_path = locations.in_resources("tests/vision/img1.jpg")
    with ImageProcessor(test_img_path) as image_processor:
        expected_metadata = {
            "width": 1000,
            "height": 666,
            "size_bytes": 110151,
            "format": "JPEG",
            "format_description": "JPEG (ISO 10918)",
            "filename": test_img_path,
        }

        assert image_processor.metadata() == expected_metadata
