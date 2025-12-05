import json
from pathlib import Path
from typing import Union

from .metrics import compute_all_metrics, normalize_caption


def evaluate_dataset(
    predictions_path: Union[str, Path],
    ground_truth_path: Union[str, Path],
):
    """
    Evaluate caption predictions against ground truth.

    predictions JSON → { "imgid": "predicted caption" }
    ground truth JSON → { "imgid": "ground truth caption" }

    Returns aggregated BLEU/METEOR/ROUGE-L scores as a dict.
    """

    predictions_path = Path(predictions_path)
    ground_truth_path = Path(ground_truth_path)

    with predictions_path.open("r") as f:
        preds = json.load(f)

    with ground_truth_path.open("r") as f:
        gts = json.load(f)

    # Find matching image ids
    image_ids = list(set(preds.keys()) & set(gts.keys()))

    if len(image_ids) == 0:
        raise ValueError("No matching image IDs between predictions & ground truth!")

    total = {
        "BLEU-1": 0.0,
        "BLEU-4": 0.0,
        "METEOR": 0.0,
        "ROUGE-L": 0.0,
    }

    for img_id in image_ids:
        pred = normalize_caption(preds[img_id])
        ref = normalize_caption(gts[img_id])

        metrics = compute_all_metrics(pred, ref)

        for k in total:
            total[k] += metrics[k]

    # Average
    count = len(image_ids)
    final_scores = {k: total[k] / count for k in total}

    return final_scores


if __name__ == "__main__":
    # Example standalone usage (optional)
    preds_file = "data/processed/predictions_val.json"
    gts_file = "data/processed/captions.json"

    scores = evaluate_dataset(preds_file, gts_file)
    print(scores)
