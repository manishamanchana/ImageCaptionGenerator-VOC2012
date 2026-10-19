import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

# Adjust if your layout is different
DATA_DIR = Path("data")
ANN_SPLITS = ["train", "val"]   # you can add "trainval" if you want

def collect_class_counts():
    counts = Counter()

    for split in ANN_SPLITS:
        ann_dir = DATA_DIR / split / "ann"
        if not ann_dir.exists():
            print(f"Warning: {ann_dir} not found, skipping.")
            continue

        for json_path in ann_dir.glob("*.jpg.json"):
            with json_path.open("r") as f:
                data = json.load(f)
            for obj in data.get("objects", []):
                label = obj.get("classTitle")
                if label:
                    counts[label] += 1

    return counts


def plot_class_distribution(counts: Counter, out_path: Path):
    labels = list(counts.keys())
    values = [counts[l] for l in labels]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of object instances")
    plt.title("Pascal VOC (DatasetNinja) class distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved class distribution figure to {out_path}")


if __name__ == "__main__":
    counts = collect_class_counts()
    print("Top 10 classes:", counts.most_common(10))

    out_path = Path("figures/class_distribution.png")
    out_path.parent.mkdir(exist_ok=True)
    plot_class_distribution(counts, out_path)
