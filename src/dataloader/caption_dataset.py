import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

from src.config import DATA_DIR, CAPTIONS_JSON, VOCAB_JSON


def simple_tokenize(text: str) -> List[str]:
    # captions we generated are very simple, so basic split is enough
    return text.lower().strip().split()


def load_vocab() -> Dict[str, int]:
    with VOCAB_JSON.open() as f:
        word2idx = json.load(f)
    return word2idx


def caption_to_indices(
    caption: str,
    word2idx: Dict[str, int],
    bos_idx: int,
    eos_idx: int,
    unk_idx: int,
) -> torch.Tensor:
    tokens = simple_tokenize(caption)
    ids: List[int] = [bos_idx]

    for tok in tokens:
        ids.append(word2idx.get(tok, unk_idx))

    ids.append(eos_idx)
    return torch.tensor(ids, dtype=torch.long)


class CaptionDataset(Dataset):
    """
    Dataset that:
      - reads images from data/{split}/img
      - reads captions from CAPTIONS_JSON ({ "split/imgid": "caption" })
      - returns transformed image tensor + caption token ids
    """

    def __init__(self, split: str = "train", transform=None, min_len: int = 1):
        self.split = split
        self.split_dir = DATA_DIR / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")

        self.img_dir = self.split_dir / "img"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.img_dir}")

        # --- load captions for this split ---
        with CAPTIONS_JSON.open() as f:
            all_caps: Dict[str, str] = json.load(f)

        # keys look like "train/2008_000001"
        self.samples: List[Dict[str, Any]] = []
        for key, cap in all_caps.items():
            split_name, img_id = key.split("/", 1)
            if split_name != split:
                continue

            img_path = self.img_dir / f"{img_id}.jpg"
            if not img_path.exists():
                # try other extensions just in case
                for ext in [".jpeg", ".png"]:
                    candidate = self.img_dir / f"{img_id}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
            if not img_path.exists():
                continue

            if len(simple_tokenize(cap)) < min_len:
                continue

            self.samples.append(
                {
                    "image_id": img_id,
                    "image_path": img_path,
                    "caption": cap,
                }
            )

        if not self.samples:
            raise RuntimeError(f"No captioned samples found for split={split}")

        self.word2idx = load_vocab()
        self.pad_idx = self.word2idx["<pad>"]
        self.bos_idx = self.word2idx["<bos>"]
        self.eos_idx = self.word2idx["<eos>"]
        self.unk_idx = self.word2idx["<unk>"]

        self.transform = transform or T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_path: Path = item["image_path"]
        caption: str = item["caption"]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        cap_ids = caption_to_indices(
            caption,
            self.word2idx,
            bos_idx=self.bos_idx,
            eos_idx=self.eos_idx,
            unk_idx=self.unk_idx,
        )

        return {
            "image_id": item["image_id"],
            "image": img_tensor,          # (3, 224, 224)
            "caption_ids": cap_ids,       # (L,)
        }


def caption_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to pad captions in a batch.

    Returns:
      images: (B, 3, 224, 224)
      captions: (B, max_len)
      lengths: (B,)
      image_ids: list[str]
      pad_idx: int
    """
    # assume all items share same pad_idx
    pad_idx = batch[0]["caption_ids"].new_tensor(
        [0]
    ).item()  # will override below anyway

    captions = [b["caption_ids"] for b in batch]
    lengths = [len(c) for c in captions]
    max_len = max(lengths)
    B = len(batch)

    # stack images
    images = torch.stack([b["image"] for b in batch], dim=0)

    # pad captions
    pad_idx = batch[0]["caption_ids"].new_tensor([0]).item()  # temp
    # we don't actually know pad_idx here, but we'll not use ignore_index from this fn
    # training loop will have real pad_idx from vocab

    padded = torch.full((B, max_len), pad_idx, dtype=torch.long)
    for i, cap in enumerate(captions):
        L = len(cap)
        padded[i, :L] = cap

    image_ids = [b["image_id"] for b in batch]

    return {
        "images": images,
        "captions": padded,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "image_ids": image_ids,
    }
