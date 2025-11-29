from pathlib import Path

# Root of the repo (…/ImageCaptionProject)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data root
DATA_DIR = PROJECT_ROOT / "data"

# Dataset splits you have
SPLITS = ["train", "val", "test", "trainval"]

# Where we'll save processed artifacts
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CAPTIONS_JSON = PROCESSED_DIR / "captions.json"
VOCAB_JSON = PROCESSED_DIR / "vocab.json"
