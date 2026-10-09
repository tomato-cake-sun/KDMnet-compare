from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PACPVCCNNLSTMConfig:
    in_channels: int = 1
    num_classes: int = 3
    input_len: int = 640
    c1: int = 128
    c2: int = 64
    c3: int = 32
    lstm_hidden: int = 32
    lstm_layers: int = 4
    fc_hidden: int = 32
    dropout: float = 0.5


class PaperPACPVCCNNLSTM(nn.Module):
    

    def __init__(self, cfg: PACPVCCNNLSTMConfig = PACPVCCNNLSTMConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        in_ch = 2 if cfg.in_channels == 1 else cfg.in_channels * 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, cfg.c1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(cfg.c1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(cfg.c1, cfg.c2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(cfg.c2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(cfg.c2, cfg.c3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(cfg.c3),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=cfg.c3,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.fc1 = nn.Linear(cfg.lstm_hidden, cfg.fc_hidden)
        self.drop = nn.Dropout(p=cfg.dropout)
        self.fc2 = nn.Linear(cfg.fc_hidden, cfg.num_classes)

    @staticmethod
    def _derivative_channel(x: torch.Tensor) -> torch.Tensor:
        dx = x[..., 1:] - x[..., :-1]
        dx = F.pad(dx, (0, 1), mode="constant", value=0.0)
        return dx

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,L) or (B,C,L), got {tuple(x.shape)}")

        x = F.adaptive_avg_pool1d(x, self.cfg.input_len)
        dx = self._derivative_channel(x)
        x = torch.cat([x, dx], dim=1)

        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        h = h.transpose(1, 2).contiguous()
        h, _ = self.lstm(h)
        h = h[:, -1, :]
        h = F.relu(self.fc1(h), inplace=True)
        h = self.drop(h)
        logits = self.fc2(h)

        if return_probs:
            return logits, torch.softmax(logits, dim=1)
        return logits


class WeightedCrossEntropyLoss(nn.Module):
    

    def __init__(self, class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None, persistent=False)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        w = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return F.cross_entropy(logits, targets.long(), weight=w, label_smoothing=self.label_smoothing)


def build_pac_pvc_cnn_lstm_model(cfg: PACPVCCNNLSTMConfig | None = None) -> PaperPACPVCCNNLSTM:
    return PaperPACPVCCNNLSTM(cfg if cfg is not None else PACPVCCNNLSTMConfig())


def build_pac_pvc_loss(
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> WeightedCrossEntropyLoss:
    return WeightedCrossEntropyLoss(class_weights=class_weights, label_smoothing=label_smoothing)


def create_training_components(num_classes: int, labels=None):
    config = PACPVCCNNLSTMConfig(in_channels=12, num_classes=num_classes, input_len=640)
    weights = None
    if labels is not None:
        counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes).clamp_min(1)
        weights = counts.sum().float() / (num_classes * counts.float())
    return PaperPACPVCCNNLSTM(config), WeightedCrossEntropyLoss(class_weights=weights)


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return x


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits = forward_logits(model, x)
    return logits, criterion(logits, y)

TRAINING_CONFIG = {"batch_size": 64, "learning_rate": 0.001, "optimizer": "Adam", "scheduler": {"type": "step", "step_size": 10, "gamma": 0.8}}

