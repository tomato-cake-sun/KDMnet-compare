from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention1D(nn.Module):
    

    def __init__(self, channels: int = 512, reduction: int = 8) -> None:
        super().__init__()
        hidden = channels // reduction
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x).squeeze(-1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1)


class ConvStage1D(nn.Module):
    

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        use_pool: bool,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        ]
        if use_pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class CNNAttentionConfig:
    in_channels: int = 1
    num_classes: int = 5
    conv_dropout: float = 0.2
    cls_dropout: float = 0.5
    attention_reduction: int = 8
    k1: int = 7
    k2: int = 5
    k3: int = 3
    k4: int = 3


class PaperCNNAttention1D(nn.Module):
    

    def __init__(self, cfg: CNNAttentionConfig = CNNAttentionConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage1 = ConvStage1D(cfg.in_channels, 64, cfg.k1, use_pool=False, dropout=cfg.conv_dropout)
        self.stage2 = ConvStage1D(64, 128, cfg.k2, use_pool=True, dropout=cfg.conv_dropout)
        self.stage3 = ConvStage1D(128, 256, cfg.k3, use_pool=True, dropout=cfg.conv_dropout)
        self.stage4 = ConvStage1D(256, 512, cfg.k4, use_pool=True, dropout=cfg.conv_dropout)

        self.attn = SEAttention1D(channels=512, reduction=cfg.attention_reduction)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.drop = nn.Dropout(p=cfg.cls_dropout)
        self.fc2 = nn.Linear(256, cfg.num_classes)

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,L) or (B,C,L), got {tuple(x.shape)}")

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.attn(x)
        h = self.gap(x).squeeze(-1)
        h = F.relu(self.fc1(h), inplace=True)
        h = self.drop(h)
        logits = self.fc2(h)

        if return_probs:
            probs = torch.softmax(logits, dim=1)
            return logits, probs
        return logits


class FocalL2Loss(nn.Module):
    

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        l2_lambda: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.l2_lambda = l2_lambda
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        ce = F.cross_entropy(logits, targets.long(), reduction="none")
        pt = torch.exp(-ce)
        fl = self.alpha * ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            loss = fl.mean()
        elif self.reduction == "sum":
            loss = fl.sum()
        else:
            loss = fl

        if model is not None and self.l2_lambda > 0:
            l2 = torch.zeros((), device=logits.device, dtype=logits.dtype)
            for p in model.parameters():
                l2 = l2 + p.pow(2).sum()
            loss = loss + self.l2_lambda * l2
        return loss


def build_cnn_attention_model(cfg: CNNAttentionConfig | None = None) -> PaperCNNAttention1D:
    return PaperCNNAttention1D(cfg if cfg is not None else CNNAttentionConfig())


def build_cnn_attention_loss(
    alpha: float = 1.0,
    gamma: float = 2.0,
    l2_lambda: float = 0.0,
    reduction: str = "mean",
) -> FocalL2Loss:
    return FocalL2Loss(alpha=alpha, gamma=gamma, l2_lambda=l2_lambda, reduction=reduction)


def create_training_components(num_classes: int, labels=None):
    config = CNNAttentionConfig(in_channels=12, num_classes=num_classes)
    return PaperCNNAttention1D(config), FocalL2Loss(alpha=1.0, gamma=2.0, l2_lambda=0.01)


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return x


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits = forward_logits(model, x)
    return logits, criterion(logits, y, model)

TRAINING_CONFIG = {"batch_size": 32, "learning_rate": 0.001, "optimizer": "Adam", "scheduler": {"type": "plateau", "mode": "min", "factor": 0.5, "patience": 3}, "weight_decay": 0.01}
