import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torchvision import transforms

from src.config import VOCAB_JSON
from src.models.encoder import EncoderCNN
from src.models.decoder import DecoderRNN
from src.dataloader.caption_dataset import CaptionDataset, caption_collate_fn


def load_vocab_info():
    with VOCAB_JSON.open() as f:
        word2idx = json.load(f)
    pad_idx = word2idx["<pad>"]
    bos_idx = word2idx["<bos>"]
    eos_idx = word2idx["<eos>"]
    return word2idx, pad_idx, bos_idx, eos_idx


def get_dataloaders(batch_size: int = 32):
    encoder_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = CaptionDataset(split="train", transform=encoder_transform)
    val_ds   = CaptionDataset(split="val",   transform=encoder_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=caption_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=caption_collate_fn,
    )

    return train_loader, val_loader


def train_one_epoch(
    encoder,
    decoder,
    loader,
    optimizer,
    criterion,
    device,
    pad_idx,
    epoch: int = 0,
):
    encoder.train()
    decoder.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        captions = batch["captions"].to(device)

        # Teacher forcing setup:
        #  - input to decoder: all tokens except last
        #  - target for loss:  all tokens except first
        inputs  = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()

        # Forward pass
        features = encoder(images)              # (B, feat_dim)
        logits   = decoder(features, inputs)    # (B, T-1, vocab_size)

        B, Tm1, V = logits.shape
        loss = criterion(
            logits.reshape(B * Tm1, V),
            targets.reshape(B * Tm1),
        )

        loss.backward()
        clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  [epoch {epoch} batch {batch_idx}] loss = {loss.item():.4f}")

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


@torch.no_grad()
def validate(
    encoder,
    decoder,
    loader,
    criterion,
    device,
    pad_idx,
    epoch: int = 0,
):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["images"].to(device)
        captions = batch["captions"].to(device)

        inputs  = captions[:, :-1]
        targets = captions[:, 1:]

        features = encoder(images)
        logits   = decoder(features, inputs)

        B, Tm1, V = logits.shape
        loss = criterion(
            logits.reshape(B * Tm1, V),
            targets.reshape(B * Tm1),
        )

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    print(f"[val   epoch {epoch}] loss = {avg_loss:.4f}")
    return avg_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    word2idx, pad_idx, bos_idx, eos_idx = load_vocab_info()
    vocab_size = len(word2idx)
    print("Vocab size:", vocab_size)

    train_loader, val_loader = get_dataloaders(batch_size=32)

    # Models
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

    # Loss & optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(
        list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters()),
        lr=1e-3,
    )

    num_epochs = 5
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        train_loss = train_one_epoch(
            encoder, decoder, train_loader, optimizer, criterion, device, pad_idx, epoch
        )
        print(f"[train epoch {epoch}] loss = {train_loss:.4f}")

        val_loss = validate(
            encoder, decoder, val_loader, criterion, device, pad_idx, epoch
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = Path("checkpoints")
            ckpt_path.mkdir(exist_ok=True)
            save_file = ckpt_path / f"best_epoch{epoch}.pt"
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "word2idx": word2idx,
                },
                save_file,
            )
            print(f"  >> Saved new best model to {save_file}")


if __name__ == "__main__":
    main()
