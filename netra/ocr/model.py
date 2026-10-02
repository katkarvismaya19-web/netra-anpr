"""
netra.ocr.model
===============

CRNN - Convolutional Recurrent Neural Network for reading plate text.

Architecture (input: 1 x 32 x 128 grayscale crop)::

    ┌──────────────┐   ┌─────────────────┐   ┌─────────────┐   ┌──────────┐
    │  CNN blocks  │ → │ collapse height │ → │ 2 x BiLSTM  │ → │  Linear  │ → CTC
    │ (features)   │   │ to a sequence   │   │ (context)   │   │ (37 cls) │
    └──────────────┘   └─────────────────┘   └─────────────┘   └──────────┘

* The **CNN** learns visual features of strokes and characters. Pooling
  shrinks the height much faster than the width, so the output keeps 32
  horizontal positions - one "time step" per narrow vertical slice.
* The **BiLSTM** reads those slices left-to-right and right-to-left, so each
  prediction can use context from neighbouring characters.
* **CTC loss** lets us train with only the plate string as the label - we
  never need to mark where each individual character is located.

Reference: Shi, Bai & Yao, "An End-to-End Trainable Neural Network for
Image-based Sequence Recognition" (TPAMI 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from netra.ocr.charset import NUM_CLASSES


def _conv_block(in_ch: int, out_ch: int, batch_norm: bool = True) -> list[nn.Module]:
    """3x3 convolution -> (optional) batch norm -> ReLU."""
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=not batch_norm)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return layers


class CRNN(nn.Module):
    """CNN feature extractor + bidirectional LSTM sequence model."""

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_size: int = 256) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            *_conv_block(1, 64),    nn.MaxPool2d(2, 2),          # 32x128 -> 16x64
            *_conv_block(64, 128),  nn.MaxPool2d(2, 2),          # 16x64  -> 8x32
            *_conv_block(128, 256),
            *_conv_block(256, 256), nn.MaxPool2d((2, 1), (2, 1)),  # 8x32 -> 4x32
            *_conv_block(256, 512),
            *_conv_block(512, 512), nn.MaxPool2d((2, 1), (2, 1)),  # 4x32 -> 2x32
            nn.Dropout2d(0.1),
        )
        # Average the remaining height so each column becomes one feature vector.
        self.collapse = nn.AdaptiveAvgPool2d((1, None))          # 2x32 -> 1x32

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: batch of images, shape (B, 1, 32, 128), values in [0, 1].

        Returns:
            Log-probabilities of shape (T, B, num_classes) with T = 32,
            the layout expected by ``nn.CTCLoss``.
        """
        features = self.collapse(self.cnn(x))          # (B, 512, 1, T)
        features = features.squeeze(2).permute(2, 0, 1)  # (T, B, 512)
        sequence, _ = self.rnn(features)               # (T, B, 2*hidden)
        logits = self.classifier(sequence)             # (T, B, classes)
        return logits.log_softmax(dim=2)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable weights - useful for the report and dashboard."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
