from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperCNN1D(nn.Module):
    

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=80, stride=11, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 32, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(32)

        self.conv5 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.drop5 = nn.Dropout(p=dropout)
        self.bn5 = nn.BatchNorm1d(32)

        self.conv6 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool6 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.drop6 = nn.Dropout(p=dropout)
        self.bn6 = nn.BatchNorm1d(32)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.drop2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.bn3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.bn4(x)

        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = self.drop5(x)
        x = self.bn5(x)

        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        x = self.drop6(x)
        x = self.bn6(x)

        x = self.gap(x).flatten(1)
        logits = self.fc(x)

        if return_probs:
            probs = torch.softmax(logits, dim=1)
            return logits, probs
        return logits


class CategoricalCrossEntropyLoss(nn.Module):
    

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        return self.ce(logits, targets.long())


def build_paper_model(
    in_channels: int = 1,
    num_classes: int = 4,
    dropout: float = 0.3,
) -> PaperCNN1D:
    return PaperCNN1D(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )


def build_paper_loss(label_smoothing: float = 0.0) -> CategoricalCrossEntropyLoss:
    return CategoricalCrossEntropyLoss(label_smoothing=label_smoothing)


def create_training_components(num_classes: int, labels=None):
    return PaperCNN1D(in_channels=12, num_classes=num_classes, dropout=0.3), CategoricalCrossEntropyLoss()


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return x


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits = forward_logits(model, x)
    return logits, criterion(logits, y)

TRAINING_CONFIG = {"batch_size": 128, "learning_rate": 0.002, "optimizer": "Adam", "scheduler": {"type": "exponential", "gamma": 0.96}}
