import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.models.encoder import EncoderCNN
from src.models.decoder import DecoderRNN
from src.dataloader.caption_dataset import CaptionDataset, caption_collate_fn
from src.eval.evaluate import evaluate_dataset  # teammate's code


# ----------------- helpers ----------------- #

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load trained encoder, decoder and vocab info from a checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    word2idx: Dict[str, int] = ckpt["word2idx"]
    vocab_size = len(word2idx)

    pad_idx = word2idx["<pad>"]
    bos_idx = word2idx["<bos>"]
    eos_idx = word2idx["<eos>"]

    # build idx2word for decoding
    idx2word = {idx: w for w, idx in word2idx.items()}

    # recreate models with same hyperparameters as training
    encoder = EncoderCNN(embed_size=256, train_cnn=False).to(device)
    decoder = DecoderRNN(
        vocab_size=vocab_size,
        embed_size=256,
        hidden_size=512,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        feat_dim=256,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, word2idx, idx2word, pad_idx, bos_idx, eos_idx


def ids_to_caption(
    token_ids: torch.Tensor,
    idx2word: Dict[int, str],
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
) -> str:
    """
    Convert a sequence of token IDs into a human-readable caption.
    Stops at <eos>, skips <bos> and <pad>.
    """
    words = []
    for idx in token_ids.tolist():
        if idx == eos_idx:
            break
        if idx in (bos_idx, pad_idx):
            continue
        word = idx2word.get(idx, "<unk>")
        words.append(word)
    if not words:
        return "<empty>"
    return " ".join(words)


def run_inference(
    split: str,
    checkpoint_path: str,
    predictions_path: str,
    max_len: int = 20,
    batch_size: int = 32,
):
    """
    Generate captions for all images in the given split ('train', 'val', 'test').

    Saves predictions in JSON format:
        { "split/imgid": "generated caption", ... }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder, decoder, word2idx, idx2word, pad_idx, bos_idx, eos_idx = load_checkpoint(
        checkpoint_path, device
    )

    # dataset will load images + (unused) ground-truth captions
    dataset = CaptionDataset(split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=caption_collate_fn,
    )

    preds: Dict[str, str] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["images"].to(device)
            image_ids = batch["image_ids"]

            # (B, feat_dim)
            features = encoder(images)

            # (B, T) predicted token IDs
            pred_ids = decoder.generate(features, max_len=max_len)
            pred_ids = pred_ids.cpu()

            for img_id, seq in zip(image_ids, pred_ids):
                caption = ids_to_caption(seq, idx2word, bos_idx, eos_idx, pad_idx)
                key = f"{split}/{img_id}"
                preds[key] = caption

            if batch_idx % 20 == 0:
                print(f"  processed batch {batch_idx}/{len(loader)}")

    # save predictions
    out_path = Path(predictions_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(preds, f, indent=2)

    print(f"Saved {len(preds)} predictions to {out_path}")
    return out_path


# ----------------- main entry ----------------- #

def main():
    # choose which split to evaluate on: 'val' or 'test'
    split = "val"

    # path to your best checkpoint (update the filename if needed)
    checkpoint_path = "checkpoints/best_epoch4.pt"

    # where to store generated captions
    predictions_path = f"data/processed/predictions_{split}.json"

    # 1) run inference
    pred_path = run_inference(
        split=split,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        max_len=20,
        batch_size=32,
    )

    # 2) evaluate against ground truth captions
    gts_path = "data/processed/captions.json"
    scores = evaluate_dataset(str(pred_path), gts_path)

    print("\n=== Evaluation scores ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
