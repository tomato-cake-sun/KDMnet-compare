from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    
    sq_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * x / torch.sqrt(sq_norm + eps)


class DeepEnsembleCNNRNN(nn.Module):
    

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        lstm_hidden_size: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        c1, c2, c3 = conv_channels
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.bi_lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv_blocks(x)
        feats = feats.mean(dim=2)
        feats = feats.transpose(1, 2).contiguous()
        seq_features, _ = self.bi_lstm(feats)
        return seq_features


class NGramConvLayer(nn.Module):
    

    def __init__(
        self,
        in_dim: int,
        out_channels_per_kernel: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        pool_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_dim,
                    out_channels=out_channels_per_kernel,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        self.out_channels = out_channels_per_kernel * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        feats = [self.pool(F.relu(conv(x))) for conv in self.convs]
        return torch.cat(feats, dim=1)


class PrimaryCapsules1D(nn.Module):
    

    def __init__(
        self,
        in_channels: int,
        num_capsules: int = 8,
        caps_dim: int = 16,
        kernel_size: int = 9,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.num_capsules = num_capsules
        self.caps_dim = caps_dim
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_capsules * caps_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.conv(x)
        bsz, _, t = u.shape
        u = u.view(bsz, self.num_capsules, self.caps_dim, t)
        u = u.permute(0, 3, 1, 2).contiguous()
        u = u.view(bsz, -1, self.caps_dim)
        return squash(u, dim=-1)


class ClassCapsules(nn.Module):
    

    def __init__(
        self,
        num_in_capsules: int,
        in_dim: int,
        num_classes: int = 5,
        out_dim: int = 16,
        routing_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_in_capsules = num_in_capsules
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.routing_iters = routing_iters
        self.W = nn.Parameter(
            0.01
            * torch.randn(1, num_in_capsules, num_classes, out_dim, in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = x.unsqueeze(2).unsqueeze(-1)
        W = self.W.expand(bsz, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x).squeeze(-1)

        b = torch.zeros(
            bsz,
            self.num_in_capsules,
            self.num_classes,
            device=x.device,
            dtype=x.dtype,
        )
        for i in range(self.routing_iters):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
            v = squash(s, dim=-1)
            if i < self.routing_iters - 1:
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement
        return v


@dataclass
class BiCapsNetConfig:
    in_channels: int = 1
    num_classes: int = 5
    lstm_hidden_size: int = 128
    dropout: float = 0.5
    ngram_kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    ngram_out_per_kernel: int = 64
    primary_num_capsules: int = 8
    primary_caps_dim: int = 16
    class_caps_dim: int = 16
    routing_iters: int = 3


class DeepBiCapsNet(nn.Module):
    

    def __init__(self, cfg: BiCapsNetConfig = BiCapsNetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = DeepEnsembleCNNRNN(
            in_channels=cfg.in_channels,
            lstm_hidden_size=cfg.lstm_hidden_size,
            dropout=cfg.dropout,
        )
        lstm_out_dim = 2 * cfg.lstm_hidden_size
        self.ngram = NGramConvLayer(
            in_dim=lstm_out_dim,
            out_channels_per_kernel=cfg.ngram_out_per_kernel,
            kernel_sizes=cfg.ngram_kernel_sizes,
        )
        self.primary_caps = PrimaryCapsules1D(
            in_channels=self.ngram.out_channels,
            num_capsules=cfg.primary_num_capsules,
            caps_dim=cfg.primary_caps_dim,
        )
        self.class_caps: ClassCapsules | None = None

    def _build_class_caps_if_needed(self, num_in_capsules: int, ref_tensor: torch.Tensor) -> None:
        if self.class_caps is None:
            self.class_caps = ClassCapsules(
                num_in_capsules=num_in_capsules,
                in_dim=self.cfg.primary_caps_dim,
                num_classes=self.cfg.num_classes,
                out_dim=self.cfg.class_caps_dim,
                routing_iters=self.cfg.routing_iters,
            )
            self.class_caps = self.class_caps.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            if next(self.class_caps.parameters()).device != ref_tensor.device:
                self.class_caps = self.class_caps.to(device=ref_tensor.device)

    def forward(self, x: torch.Tensor):
        seq = self.feature_extractor(x)
        ng = self.ngram(seq)
        prim = self.primary_caps(ng)
        self._build_class_caps_if_needed(prim.size(1), prim)
        assert self.class_caps is not None
        class_caps = self.class_caps(prim)
        lengths = torch.norm(class_caps, dim=-1)
        return {
            "capsules": class_caps,
            "logits": lengths,
            "probs": F.softmax(lengths, dim=-1),
        }


class CapsuleMarginL2Loss(nn.Module):
    

    def __init__(
        self,
        m_plus: float = 0.9,
        m_minus: float = 0.1,
        absent_class_weight: float = 0.5,
        l2_lambda: float = 1e-5,
    ) -> None:
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.absent_class_weight = absent_class_weight
        self.l2_lambda = l2_lambda

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        if target.ndim == 1:
            t = F.one_hot(target.long(), num_classes=logits.size(1)).float()
        else:
            t = target.float()

        left = F.relu(self.m_plus - logits) ** 2
        right = F.relu(logits - self.m_minus) ** 2
        margin = t * left + self.absent_class_weight * (1.0 - t) * right
        margin_loss = margin.sum(dim=1).mean()

        if model is None or self.l2_lambda <= 0.0:
            return margin_loss

        l2_reg = torch.zeros((), device=logits.device, dtype=logits.dtype)
        for p in model.parameters():
            l2_reg = l2_reg + p.pow(2).sum()
        return margin_loss + self.l2_lambda * l2_reg


def build_bicapsnet_model(cfg: BiCapsNetConfig | None = None) -> DeepBiCapsNet:
    return DeepBiCapsNet(cfg if cfg is not None else BiCapsNetConfig())


def build_bicapsnet_loss(
    m_plus: float = 0.9,
    m_minus: float = 0.1,
    absent_class_weight: float = 0.5,
    l2_lambda: float = 1e-5,
) -> CapsuleMarginL2Loss:
    return CapsuleMarginL2Loss(
        m_plus=m_plus,
        m_minus=m_minus,
        absent_class_weight=absent_class_weight,
        l2_lambda=l2_lambda,
    )


def create_training_components(num_classes: int, labels=None):
    config = BiCapsNetConfig(in_channels=1, num_classes=num_classes)
    return DeepBiCapsNet(config), CapsuleMarginL2Loss(m_plus=0.9, m_minus=0.1, absent_class_weight=0.5, l2_lambda=1e-5)


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x.unsqueeze(1), size=(12, 500), mode="bilinear", align_corners=False)


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))["logits"]


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits = forward_logits(model, x)
    return logits, criterion(logits, y, model)

TRAINING_CONFIG = {"batch_size": 16, "learning_rate": 1e-3, "optimizer": "Adam", "scheduler": {"type": "step", "step_size": 10, "gamma": 0.8}}
