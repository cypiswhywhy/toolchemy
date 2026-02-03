import os
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from toolchemy.utils.locations import Locations


@dataclass
class Caption:
    text: str
    y: int
    font: ImageFont.FreeTypeFont | None = None
    font_name: str | None = None
    size: int | None = None
    color: tuple | None = None

    def __post_init__(self):
        if self.font_name is None:
            self.font_name = "Pacifico.ttf"
        if self.size is None:
            self.size = 60
        self.font = ImageFont.truetype(self.font_name, self.size)
        if self.color is None:
            self.color = (255, 255, 255, 255)



def add_caption(input_img_path: str, captions: list[Caption], output_img_path: str | None = None):
    # fonts = ["Pacifico.ttf", "Anton.ttf"]

    if output_img_path is None:
        path_root, path_ext = os.path.splitext(input_img_path)
        output_img_path = f"{path_root}_out{path_ext}"

    img = Image.open(input_img_path).convert("RGBA")

    txt = Image.new("RGBA", img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)

    x_start = 50
    for i, caption in enumerate(captions):
        draw.text((x_start, caption.y), caption.text, font=caption.font, fill=caption.color)

    glow = txt.filter(ImageFilter.GaussianBlur(5))

    out = Image.alpha_composite(img.convert("RGBA"), glow)
    out = Image.alpha_composite(out, txt)

    out.convert("RGB").save(output_img_path)


def main():
    std_text = "*bilet uprawnia do jednej soboty lub niedzieli przed kompem, czy coś :)"
    std_y_pos = 1900
    std_font_size = 45
    locations = Locations()
    font_name_pacifico = "/path/to/Pacifico-Regular.ttf"
    font_name_open_sans_italic = "/path/to/OpenSans-Italic-VariableFont_wdth,wght.ttf"
    bottom_font_name = font_name_open_sans_italic
    month_font_name = font_name_open_sans_italic
    add_caption(locations.in_data("kima1.jpg"), captions=[
        Caption(text="Lorem ipsum", y=30, size=100, font_name=font_name_pacifico),
        Caption(text="dolor sit amet, consectetur adipiscing elit, sed do eiusmod", y=210, size=90, font_name=font_name_pacifico),
        Caption(text="tempor incididunt ut labore et dolore magna aliqua.", y=390, size=60, font_name=month_font_name),
        Caption(text=std_text, y=std_y_pos-300, size=std_font_size, font_name=bottom_font_name),
    ])
    add_caption(locations.in_data("kima2.jpg"), captions=[
        Caption(text="Ut enim ad minim veniam,", y=30, size=100, font_name=font_name_pacifico),
        Caption(text="quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.", y=210, size=90,
                font_name=font_name_pacifico),
        Caption(text=std_text, y=std_y_pos, size=std_font_size, font_name=bottom_font_name),
    ])


if __name__ == "__main__":
    main()
