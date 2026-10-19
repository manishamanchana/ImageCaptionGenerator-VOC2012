import json
import random
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

# ---- paths ----
PREDICTIONS_JSON = Path("data/processed/predictions_val.json")
GT_CAPTIONS_JSON = Path("data/processed/captions.json")   # optional: for ground-truth
VAL_IMG_DIR = Path("data/val/img")                        # DatasetNinja layout

# ---- load predictions ----
with PREDICTIONS_JSON.open("r") as f:
    preds = json.load(f)

# (optional) load ground-truth captions to show under prediction
gt_caps = {}
if GT_CAPTIONS_JSON.exists():
    with GT_CAPTIONS_JSON.open("r") as f:
        gt_caps = json.load(f)

# we only want keys for the 'val' split
val_keys = [k for k in preds.keys() if k.startswith("val/")]

# pick 4 examples (you can change 4 -> 3 if you only want three)
num_examples = 4
example_keys = random.sample(val_keys, num_examples)

print("Chosen examples:")
for k in example_keys:
    print(" ", k, "->", preds[k])

# ---- build figure ----
cols = 2
rows = (num_examples + 1) // 2

plt.figure(figsize=(10, 8))

for i, key in enumerate(example_keys, start=1):
    split, img_id = key.split("/")     # key format: "val/2008_000123"
    img_path = VAL_IMG_DIR / f"{img_id}.jpg"

    if not img_path.exists():
        # try alternate extensions if needed
        for ext in [".jpeg", ".png"]:
            alt = VAL_IMG_DIR / f"{img_id}{ext}"
            if alt.exists():
                img_path = alt
                break

    img = Image.open(img_path).convert("RGB")

    pred_caption = preds[key]
    gt_caption = gt_caps.get(key, None)

    plt.subplot(rows, cols, i)
    plt.imshow(img)
    plt.axis("off")

    if gt_caption is not None:
        title = f"Pred: {pred_caption}\nGT:   {gt_caption}"
    else:
        title = f"Pred: {pred_caption}"

    # small font so it fits
    plt.title(title, fontsize=9)

plt.tight_layout()
out_path = Path("figures/example_captions_grid.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to {out_path.resolve()}")
plt.show()
