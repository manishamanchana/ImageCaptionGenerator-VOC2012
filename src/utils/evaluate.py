import json
from .metrics import compute_all_metrics, normalize_caption

def evaluate_dataset(predictions_path: str, ground_truth_path: str):
    """
    predictions JSON → { "imgid": "predected caption" }
    ground truth JSON → { "imgid": "gt caption" }

    Returns aggregated BLEU/METEOR/ROUGE-L scores.
    """

    with open(predictions_path, "r") as f:
        preds = json.load(f)

    with open(ground_truth_path, "r") as f:
        gts = json.load(f)

    # Find matching image ids
    image_ids = list(set(preds.keys()) & set(gts.keys()))

    if len(image_ids) == 0:
        raise ValueError("No matching image IDs between predictions & ground truth!")

    total = {
        "BLEU-1": 0.0,
        "BLEU-4": 0.0,
        "METEOR": 0.0,
        "ROUGE-L": 0.0
    }

    for img_id in image_ids:
        pred = normalize_caption(preds[img_id])
        ref  = normalize_caption(gts[img_id])

        metrics = compute_all_metrics(pred, ref)

        for k in total:
            total[k] += metrics[k]

    # Average
    count = len(image_ids)
    final_scores = {k: total[k] / count for k in total}

    return final_scores


if __name__ == "__main__":
    # Example usage
    preds_file = ""# place our predictions file path here#
    gts_file = r"data\processed\captions.json"

    scores = evaluate_dataset(preds_file, gts_file)
    print(scores)
