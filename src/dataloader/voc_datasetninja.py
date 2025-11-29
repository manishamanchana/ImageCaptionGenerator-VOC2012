import json
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.config import DATA_DIR


def parse_json_annotation(json_path: Path) -> Dict[str, Any]:
    """Parse DatasetNinja / Supervisely-style JSON annotation."""
    with json_path.open("r") as f:
        data = json.load(f)

    size = data.get("size", {})
    width = size.get("width")
    height = size.get("height")

    objects: List[str] = []
    for obj in data.get("objects", []):
        label = obj.get("classTitle")
        if label:
            objects.append(label)

    return {
        "width": width,
        "height": height,
        "objects": objects,
    }

class VOCDatasetNinja(Dataset):
    """
    Handles DatasetNinja Pascal VOC 2012 layout:

      data/{split}/ann/*.jpg.json
      data/{split}/img/*.jpg
    """

    def __init__(self, split: str = "train", transform=None, skip_empty: bool = True):
        self.split = split
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder does not exist: {split_dir}")

        self.img_dir = split_dir / "img"
        self.ann_dir = split_dir / "ann"

        if not self.img_dir.exists() or not self.ann_dir.exists():
            raise FileNotFoundError(
                f"Expected 'img' and 'ann' inside {split_dir}, "
                f"found img={self.img_dir.exists()}, ann={self.ann_dir.exists()}"
            )

        # Each json file corresponds to one image
        self.items = []
        for p in sorted(self.ann_dir.glob("*.jpg.json")):
            stem = p.stem              # "2008_000001.jpg"
            base_id = stem.replace(".jpg", "")  # "2008_000001"
            ann = parse_json_annotation(p)
            if skip_empty and not ann["objects"]:
                continue
            self.items.append((base_id, ann))

        if not self.items:
            raise RuntimeError(f"No valid annotations found in {self.ann_dir}")

        self.transform = transform or T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        base_id, ann = self.items[idx]

        img_path = self.img_dir / f"{base_id}.jpg"
        if not img_path.exists():
            for ext in [".jpeg", ".png"]:
                candidate = self.img_dir / f"{base_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for id {base_id} in {self.img_dir}")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {
            "image_id": base_id,
            "image": img,          # tensor
            "objects": ann["objects"],  # list of labels
        }
