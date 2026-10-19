import json
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Union

from .metrics import compute_all_metrics, normalize_caption

# Template endings/prefixes from caption_generator.py.
# Longer suffix first: "are present in the scene" also ends with "in the scene".
TEMPLATE_SUFFIXES = [" are present in the scene", " in the scene", " in the image"]
TEMPLATE_PREFIX = "the image contains "

COUNT_WORDS = {"one", "two", "three"}

# Plural forms that "strip a trailing s" gets wrong
IRREGULAR_NOUNS = {"people": "person", "bus": "bus", "buss": "bus"}

BLIND_CAPTION = "one person in the scene"


def caption_to_phrase(caption: str) -> str:
    """Strip the template: 'the image contains two dogs' -> 'two dogs'."""
    caption = caption.strip()
    if caption.startswith(TEMPLATE_PREFIX):
        return caption[len(TEMPLATE_PREFIX):]
    for suffix in TEMPLATE_SUFFIXES:
        if caption.endswith(suffix):
            return caption[: -len(suffix)]
    return caption


def phrase_to_parts(phrase: str) -> Counter:
    """
    'one pottedplant, one sofa, and one chair' -> Counter of
    ['one pottedplant', 'one sofa', 'one chair'].
    References use commas for 3+ classes; predictions only use 'and'.
    """
    phrase = phrase.replace(",", " and ")
    parts = [p.strip() for p in phrase.split(" and ")]
    return Counter(p for p in parts if p)


def singular(noun: str) -> str:
    if noun in IRREGULAR_NOUNS:
        return IRREGULAR_NOUNS[noun]
    if noun.endswith("s"):
        return noun[:-1]
    return noun


def parts_to_classes(parts: Counter) -> Set[str]:
    """{'three aeroplanes', 'one person'} -> {'aeroplane', 'person'}"""
    classes = set()
    for part in parts:
        words = part.split()
        if words and (words[0] in COUNT_WORDS or words[0].isdigit()):
            words = words[1:]
        classes.add(singular(" ".join(words)))
    return classes


def task_metrics(
    predictions_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
) -> Dict[str, float]:
    """
    Template-independent accuracy plus a blind baseline.

    Returns:
      exact_objects_counts: same classes with same counts
      same_class_set:       same classes, counts ignored
      blind_<metric>:       BLEU/METEOR/ROUGE-L if every prediction were BLIND_CAPTION
    """
    with Path(predictions_path).open("r") as f:
        preds = json.load(f)
    with Path(ground_truth_path).open("r") as f:
        gts = json.load(f)

    image_ids = list(set(preds.keys()) & set(gts.keys()))
    if len(image_ids) == 0:
        raise ValueError("No matching image IDs between predictions & ground truth!")

    exact = 0
    same_set = 0
    blind = {"BLEU-1": 0.0, "BLEU-4": 0.0, "METEOR": 0.0, "ROUGE-L": 0.0}

    for img_id in image_ids:
        pred_parts = phrase_to_parts(caption_to_phrase(preds[img_id]))
        ref_parts = phrase_to_parts(caption_to_phrase(gts[img_id]))

        if pred_parts == ref_parts:
            exact += 1
        if parts_to_classes(pred_parts) == parts_to_classes(ref_parts):
            same_set += 1

        scores = compute_all_metrics(
            normalize_caption(BLIND_CAPTION), normalize_caption(gts[img_id])
        )
        for k in blind:
            blind[k] += scores[k]

    count = len(image_ids)
    results = {
        "exact_objects_counts": exact / count,
        "same_class_set": same_set / count,
    }
    for k, v in blind.items():
        results[f"blind_{k}"] = v / count

    return results


if __name__ == "__main__":
    preds_file = "data/processed/predictions_val.json"
    gts_file = "data/processed/captions.json"

    scores = task_metrics(preds_file, gts_file)
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
