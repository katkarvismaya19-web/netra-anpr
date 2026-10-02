"""
Step 5 - Train the character recognition model (CRNN + CTC)
===========================================================

Trains :class:`netra.ocr.model.CRNN` on the synthetic plates (plus any real
labelled crops you add in ``data/ocr/real/``).

Training recipe
---------------
* Loss       : CTC loss - aligns the 32 output time steps with the label
               automatically, so no per-character position labels are needed.
* Optimiser  : AdamW with weight decay (L2-style regularisation).
* Schedule   : One-Cycle learning rate (warm-up, then cosine decay).
* Stability  : gradient clipping at norm 5 (LSTMs can have exploding gradients).
* Selection  : the checkpoint with the best validation *plate accuracy* is kept.

Outputs
-------
* ``models/crnn_plate.pt``         - best weights + config + metrics
* ``outputs/ocr/history.csv``      - loss / accuracy per epoch
* ``outputs/ocr/training_curves.png``

Usage
-----
    python scripts/train_ocr.py
    python scripts/train_ocr.py --epochs 10 --batch 64     # quick run
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netra.config import ensure_dir, load_config, resolve  # noqa: E402
from netra.metrics import character_error_rate, plate_accuracy  # noqa: E402
from netra.ocr.charset import BLANK, greedy_decode  # noqa: E402
from netra.ocr.dataset import PlateTextDataset, collate, worker_init  # noqa: E402
from netra.ocr.model import CRNN, count_parameters  # noqa: E402
from netra.ocr.recognizer import pick_device  # noqa: E402
from netra.postprocess import correct_plate  # noqa: E402


def set_seed(seed: int) -> None:
    """Make runs repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(cfg: dict, batch: int, workers: int):
    ocr = cfg["ocr"]
    base = resolve(cfg["paths"]["ocr_data"])
    h, w = ocr["image_height"], ocr["image_width"]

    train_sets = [PlateTextDataset([base / "synthetic" / "train"], h, w, train=True, seed=ocr["seed"])]
    val_sets = [PlateTextDataset([base / "synthetic" / "val"], h, w, train=False)]

    # Optional real crops labelled by you - oversampled x5 because they are rare and valuable.
    real_train, real_val = base / "real" / "train", base / "real" / "val"
    if (real_train / "labels.csv").exists():
        real = PlateTextDataset([real_train], h, w, train=True, seed=ocr["seed"] + 1)
        train_sets.extend([real] * 5)
        print(f"Including {len(real)} real training crops (x5 oversampling)")
    if (real_val / "labels.csv").exists():
        val_sets.append(PlateTextDataset([real_val], h, w, train=False))

    common = dict(batch_size=batch, collate_fn=collate, num_workers=workers,
                  worker_init_fn=worker_init, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(ConcatDataset(train_sets), shuffle=True, drop_last=True,
                              persistent_workers=workers > 0, **common)
    val_loader = DataLoader(ConcatDataset(val_sets), shuffle=False, **common)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    """Validation loss, CER and plate accuracy (raw and after format correction)."""
    model.eval()
    total_loss, batches = 0.0, 0
    preds, corrected, targets = [], [], []
    for images, labels, lengths, texts in loader:
        images = images.to(device)
        log_probs = model(images)
        input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
        total_loss += criterion(log_probs.cpu(), labels, input_lengths, lengths).item()
        batches += 1
        for text, _ in greedy_decode(log_probs.cpu()):
            preds.append(text)
            corrected.append(correct_plate(text).text)
        targets.extend(texts)
    return {
        "val_loss": total_loss / max(batches, 1),
        "cer": character_error_rate(preds, targets),
        "plate_acc": plate_accuracy(preds, targets),
        "plate_acc_corrected": plate_accuracy(corrected, targets),
    }


def save_curves(history: list[dict], path: Path) -> None:
    """Plot loss and accuracy curves for the project report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train", color="#1F4E9C")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation", color="#E8A317")
    ax1.set_title("CTC loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(epochs, [h["plate_acc"] * 100 for h in history], label="raw", color="#1F4E9C")
    ax2.plot(epochs, [h["plate_acc_corrected"] * 100 for h in history], label="with format correction",
             color="#2E7D4F")
    ax2.set_title("Validation plate accuracy (%)")
    ax2.set_xlabel("epoch")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    ocr = cfg["ocr"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=ocr["epochs"])
    parser.add_argument("--batch", type=int, default=ocr["batch"])
    parser.add_argument("--lr", type=float, default=ocr["learning_rate"])
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="DataLoader worker processes (use 0 on Windows if you see errors)")
    args = parser.parse_args()

    set_seed(ocr["seed"])
    device = pick_device()
    train_loader, val_loader = build_loaders(cfg, args.batch, args.workers)

    model = CRNN(hidden_size=ocr["hidden_size"]).to(device)
    print(f"Device: {device} | CRNN parameters: {count_parameters(model):,}")
    print(f"Train batches/epoch: {len(train_loader)} | val samples: {len(val_loader.dataset)}")

    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.15,
    )

    out_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "ocr")
    weights_path = resolve(ocr["weights"])
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_acc = [], -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for step, (images, labels, lengths, _) in enumerate(train_loader, 1):
            images = images.to(device)
            log_probs = model(images)                                   # (T, B, C)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long)
            # CTC on CPU tensors is deterministic and supports all devices.
            loss = criterion(log_probs.cpu(), labels, input_lengths, lengths)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        metrics = evaluate(model, val_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": running / len(train_loader), **metrics,
                  "lr": scheduler.get_last_lr()[0], "seconds": time.time() - start}
        history.append(record)
        print(f"Epoch {epoch:>3}/{args.epochs} | train {record['train_loss']:.4f} | "
              f"val {metrics['val_loss']:.4f} | CER {metrics['cer']:.4f} | "
              f"plate acc {metrics['plate_acc'] * 100:.2f}% "
              f"(corrected {metrics['plate_acc_corrected'] * 100:.2f}%) | {record['seconds']:.0f}s")

        if metrics["plate_acc"] > best_acc:
            best_acc = metrics["plate_acc"]
            torch.save({
                "model": model.state_dict(),
                "config": {k: ocr[k] for k in ("image_height", "image_width", "hidden_size")},
                "metrics": {**metrics, "epoch": epoch},
            }, weights_path)
            print(f"  saved new best model ({best_acc * 100:.2f}%) to {weights_path}")

        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        save_curves(history, out_dir / "training_curves.png")

    print(f"\nBest validation plate accuracy: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
