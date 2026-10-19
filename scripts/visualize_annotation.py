import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import base64
import zlib

ANN_PATH = Path("data/train/ann")
IMG_PATH = Path("data/train/img")

OUTPUT = "figures/sample_labeled_voc.png"

def decode_bitmap(obj):
    """Decode Supervisely bitmap mask."""
    bmp = obj["bitmap"]
    origin = bmp["origin"]
    data = zlib.decompress(base64.b64decode(bmp["data"]))
    mask = np.frombuffer(data, dtype=np.uint8)
    return mask.reshape(-1, mask.size // (mask.size // len(mask))), origin


def mask_to_bbox(mask, origin):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1 = min(xs) + origin[0]
    y1 = min(ys) + origin[1]
    x2 = max(xs) + origin[0]
    y2 = max(ys) + origin[1]
    return (x1, y1, x2, y2)


def visualize_sample(json_file):
    with open(json_file, "r") as f:
        ann = json.load(f)

    stem = json_file.stem.replace(".jpg", "")
    img = Image.open(IMG_PATH / f"{stem}.jpg").convert("RGB")
    draw = ImageDraw.Draw(img)

    for obj in ann["objects"]:
        label = obj["classTitle"]

        mask, origin = decode_bitmap(obj)
        bbox = mask_to_bbox(mask, origin)
        if bbox:
            draw.rectangle(bbox, outline="red", width=3)
            draw.text((bbox[0], bbox[1] - 10), label, fill="red")

    img.save(OUTPUT)
    print(f"Saved labeled sample image -> {OUTPUT}")


if __name__ == "__main__":
    sample_json = next(ANN_PATH.glob("*.json"))
    visualize_sample(sample_json)
