from typing import Dict, List
import json
from collections import Counter

from tqdm import tqdm

from src.config import DATA_DIR, CAPTIONS_JSON, SPLITS
from src.dataloader.voc_datasetninja import parse_json_annotation

import random

random.seed(42)


IGNORED_LABELS = {"neutral"}


TEMPLATES = [
    "{phrase} in the scene",
    "{phrase} in the image",
    "the image contains {phrase}",
    "{phrase} are present in the scene"
]



def pluralize(label: str, count: int) -> str:
    """
    Very simple pluralization:
      - 1 -> 'person'
      - 2 -> 'persons'
    You can add custom irregulars if you want later.
    """
    if count == 1:
        return label

    # simple irregulars you can expand later
    irregular = {
        "person": "people",
        "man": "men",
        "woman": "women",
        "child": "children",
    }
    if label in irregular:
        return irregular[label]

    # cheap plural rule: add 's'
    return label + "s"


def count_based_phrase(objects: List[str]) -> str:
    """
    Turn a list like ['person', 'aeroplane', 'aeroplane']
    into 'one person and two aeroplanes'.
    """
    # filter out labels we want to ignore
    filtered = [o for o in objects if o not in IGNORED_LABELS]

    if not filtered:
        return "an image"

    counts = Counter(filtered)

    parts = []
    for label, count in counts.items():
        if count == 1:
            count_word = "one"
        elif count == 2:
            count_word = "two"
        elif count == 3:
            count_word = "three"
        else:
            count_word = f"{count}"

        parts.append(f"{count_word} {pluralize(label, count)}")

    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        # 'A, B, and C' style
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def objects_to_caption(objects: List[str]) -> str:
    phrase = count_based_phrase(objects)
    template = random.choice(TEMPLATES)
    return template.format(phrase=phrase)



def generate_captions_for_split(split: str) -> Dict[str, str]:
    split_ann_dir = DATA_DIR / split / "ann"
    if not split_ann_dir.exists():
        raise FileNotFoundError(f"Annotation folder not found for split '{split}'")

    captions: Dict[str, str] = {}
    for json_path in tqdm(sorted(split_ann_dir.glob("*.jpg.json")), desc=f"{split} captions"):
        stem = json_path.stem               # "2008_000001.jpg"
        base_id = stem.replace(".jpg", "")  # "2008_000001"

        ann = parse_json_annotation(json_path)
        objs = ann["objects"]
        if not objs:
            continue  # skip images with no objects

        cap = objects_to_caption(objs)
        captions[f"{split}/{base_id}"] = cap  # key includes split

    return captions


def generate_all_captions():
    all_caps: Dict[str, str] = {}
    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        split_caps = generate_captions_for_split(split)
        all_caps.update(split_caps)

    with CAPTIONS_JSON.open("w") as f:
        json.dump(all_caps, f, indent=2)

    print(f"Saved {len(all_caps)} captions to {CAPTIONS_JSON}")


if __name__ == "__main__":
    generate_all_captions()
