from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + x, inplace=True)


class MomentumConvBlock1D(nn.Module):
    

    def __init__(self, channels: int, momentum: float = 0.9) -> None:
        super().__init__()
        self.momentum = momentum
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.momentum * x + (1.0 - self.momentum) * h


class CycleCMCHAGenerator1D(nn.Module):
    

    def __init__(self, in_channels: int = 1, base_len: int = 256) -> None:
        super().__init__()
        self.base_len = base_len
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(ResidualBlock1D(256), ResidualBlock1D(256))
        self.momentum_block = MomentumConvBlock1D(256, momentum=0.9)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv1d(128, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected (B,L) or (B,C,L), got {tuple(x.shape)}")
        if x.size(-1) != self.base_len:
            x = F.adaptive_avg_pool1d(x, self.base_len)

        h = self.conv1(x)
        h = self.conv2(h)
        h = self.down(h)
        h = self.res(h)
        h = self.momentum_block(h)
        h = self.up(h)
        y = torch.tanh(self.out_conv(h))
        return y


class CycleCMCHADiscriminator1D(nn.Module):
    

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x)


@dataclass
class CycleCMCHAClassifierConfig:
    in_channels: int = 1
    num_classes: int = 4
    input_len: int = 256
    dropout: float = 0.5


class CycleCMCHACNNClassifier(nn.Module):
    

    def __init__(self, cfg: CycleCMCHAClassifierConfig = CycleCMCHAClassifierConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(
            nn.Conv1d(cfg.in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(128)
        self.fc1 = nn.Linear(256 * 128, 512)
        self.drop = nn.Dropout(p=cfg.dropout)
        self.fc_out = nn.Linear(512, cfg.num_classes)

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected (B,L) or (B,C,L), got {tuple(x.shape)}")
        if x.size(-1) != self.cfg.input_len:
            x = F.adaptive_avg_pool1d(x, self.cfg.input_len)

        h = self.conv1(x)
        h = self.conv2(h)
        h = self.project(h)
        h = self.temporal_pool(h)
        h = h.flatten(1)
        h = F.relu(self.fc1(h), inplace=True)
        h = self.drop(h)
        logits = self.fc_out(h)
        if return_probs:
            probs = torch.softmax(logits, dim=1)
            return logits, probs
        return logits


class CompositeCycleCMCHALoss(nn.Module):
    

    def __init__(
        self,
        w_cls: float = 1.0,
        w_cycle: float = 10.0,
        w_adv: float = 1.0,
        w_guided: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_cls = w_cls
        self.w_cycle = w_cycle
        self.w_adv = w_adv
        self.w_guided = w_guided
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        cycle_loss: Optional[torch.Tensor] = None,
        adv_loss_e: Optional[torch.Tensor] = None,
        adv_loss_n: Optional[torch.Tensor] = None,
        guided_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        cls = self.ce(logits, targets.long())

        total = self.w_cls * cls
        if cycle_loss is not None:
            total = total + self.w_cycle * cycle_loss
        if adv_loss_e is not None:
            total = total + self.w_adv * adv_loss_e
        if adv_loss_n is not None:
            total = total + self.w_adv * adv_loss_n
        if guided_loss is not None:
            total = total + self.w_guided * guided_loss
        return total


def cycle_consistency_l1(real_x: torch.Tensor, rec_x: torch.Tensor, real_y: torch.Tensor, rec_y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(rec_x, real_x) + F.l1_loss(rec_y, real_y)


def adversarial_generator_lsgan(pred_fake: torch.Tensor) -> torch.Tensor:
    target_real = torch.ones_like(pred_fake)
    return F.mse_loss(pred_fake, target_real)


def adversarial_discriminator_lsgan(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    target_real = torch.ones_like(pred_real)
    target_fake = torch.zeros_like(pred_fake)
    return 0.5 * (F.mse_loss(pred_real, target_real) + F.mse_loss(pred_fake, target_fake))


def guided_regularization(fake_signal: torch.Tensor, ref_signal: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(fake_signal, ref_signal)


def build_cycle_cmcha_classifier(cfg: Optional[CycleCMCHAClassifierConfig] = None) -> CycleCMCHACNNClassifier:
    return CycleCMCHACNNClassifier(cfg if cfg is not None else CycleCMCHAClassifierConfig())


def build_cycle_cmcha_gan(
    in_channels: int = 1,
    input_len: int = 256,
) -> tuple[CycleCMCHAGenerator1D, CycleCMCHAGenerator1D, CycleCMCHADiscriminator1D, CycleCMCHADiscriminator1D]:
    gen_e = CycleCMCHAGenerator1D(in_channels=in_channels, base_len=input_len)
    gen_n = CycleCMCHAGenerator1D(in_channels=in_channels, base_len=input_len)
    dis_x = CycleCMCHADiscriminator1D(in_channels=in_channels)
    dis_y = CycleCMCHADiscriminator1D(in_channels=in_channels)
    return gen_e, gen_n, dis_x, dis_y


def build_cycle_cmcha_loss(
    w_cls: float = 1.0,
    w_cycle: float = 10.0,
    w_adv: float = 1.0,
    w_guided: float = 1.0,
    label_smoothing: float = 0.0,
) -> CompositeCycleCMCHALoss:
    return CompositeCycleCMCHALoss(
        w_cls=w_cls,
        w_cycle=w_cycle,
        w_adv=w_adv,
        w_guided=w_guided,
        label_smoothing=label_smoothing,
    )


def create_training_components(num_classes: int, labels=None):
    config = CycleCMCHAClassifierConfig(in_channels=12, num_classes=num_classes, input_len=256, dropout=0.5)
    return CycleCMCHACNNClassifier(config), CompositeCycleCMCHALoss(w_cls=1.0)


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return x


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits = forward_logits(model, x)
    return logits, criterion(logits, y)

TRAINING_CONFIG = {"batch_size": 64, "learning_rate": 0.001, "optimizer": "Adam", "scheduler": {"type": "exponential", "gamma": 0.96}}

