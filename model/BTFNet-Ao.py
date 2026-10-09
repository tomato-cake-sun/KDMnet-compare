from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeTF(nn.Module):
    

    def __init__(self, max_pyramid_scale: int = 16) -> None:
        super().__init__()
        self.max_pyramid_scale = max_pyramid_scale
        self.scales = [k for k in range(2, max_pyramid_scale + 1, 2)]
        self.alpha = nn.Parameter(torch.ones(len(self.scales)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_fft = torch.fft.rfft(x, dim=-1)
        mag = torch.abs(x_fft)

        ms = []
        f = mag.size(-1)
        for i, k in enumerate(self.scales):
            p = F.max_pool1d(mag, kernel_size=min(k, f), stride=min(k, f), ceil_mode=False)
            up = F.interpolate(p, size=f, mode="linear", align_corners=False)
            ms.append(self.alpha[i].abs() * up)
        if len(ms) == 0:
            mag_enh = mag
        else:
            mag_enh = (mag + sum(ms)) / (len(ms) + 1.0)
        phase = torch.angle(x_fft)
        x_fft_enh = torch.polar(mag_enh, phase)
        x_time_enh = torch.fft.irfft(x_fft_enh, n=x.size(-1), dim=-1)
        x_freq = torch.cat([x_fft_enh.real, x_fft_enh.imag], dim=1)
        return x_time_enh, x_freq


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, ch: int, drop: float = 0.1) -> None:
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, kernel_size=7, padding=3, groups=ch)
        self.pw1 = nn.Conv1d(ch, 4 * ch, kernel_size=1)
        self.pw2 = nn.Conv1d(4 * ch, ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = F.gelu(self.pw1(h))
        h = self.drop(self.pw2(h))
        return self.bn(x + h)


class LAEConvNeXt(nn.Module):
    

    def __init__(self, in_ch: int, out_dim: int = 512) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=3, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=3, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(ConvNeXtBlock1D(16), ConvNeXtBlock1D(16), nn.MaxPool1d(2))
        self.down = nn.Conv1d(16, 64, kernel_size=1)
        self.block2 = nn.Sequential(ConvNeXtBlock1D(64), ConvNeXtBlock1D(64), nn.MaxPool1d(2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.down(h)
        h = self.block2(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class Residual1DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class LAEResNet1D(nn.Module):
    

    def __init__(self, in_ch: int, out_dim: int = 512) -> None:
        super().__init__()
        self.stem = nn.Conv1d(in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.b0 = Residual1DBlock(64, 64, stride=1)
        self.b1 = Residual1DBlock(64, 64, stride=2)
        self.b2 = Residual1DBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.stem(x), inplace=True)
        h = self.b0(h)
        h = self.b1(h)
        h = self.b2(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class GAETransformer(nn.Module):
    

    def __init__(
        self,
        in_ch: int,
        seq_len: int,
        patch_size: int = 6,
        embed_dim: int = 512,
        depth: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 512,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = max(1, seq_len // patch_size)

        self.patch_proj = nn.Conv1d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch_proj(x).transpose(1, 2).contiguous()
        b = t.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, t], dim=1)
        pos = self.pos[:, : tokens.size(1), :]
        h = self.encoder(tokens + pos)
        return self.norm(h[:, 0, :])


class MultiViewMutualFusion(nn.Module):
    

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.qg = nn.Linear(dim, dim, bias=False)
        self.kg = nn.Linear(dim, dim, bias=False)
        self.vg = nn.Linear(dim, dim, bias=False)
        self.ql = nn.Linear(dim, dim, bias=False)
        self.kl = nn.Linear(dim, dim, bias=False)
        self.vl = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, local: torch.Tensor, global_: torch.Tensor) -> torch.Tensor:
        qg = self.qg(global_).unsqueeze(1)
        kl = self.kl(local).unsqueeze(1)
        vl = self.vl(local).unsqueeze(1)
        l2g = torch.softmax(torch.matmul(qg, kl.transpose(1, 2)) / (local.size(1) ** 0.5), dim=-1)
        l2g = torch.matmul(l2g, vl).squeeze(1)

        ql = self.ql(local).unsqueeze(1)
        kg = self.kg(global_).unsqueeze(1)
        vg = self.vg(global_).unsqueeze(1)
        g2l = torch.softmax(torch.matmul(ql, kg.transpose(1, 2)) / (local.size(1) ** 0.5), dim=-1)
        g2l = torch.matmul(g2l, vg).squeeze(1)

        return self.out(torch.cat([l2g, g2l], dim=1))


@dataclass
class BTFNetConfig:
    in_channels: int = 1
    input_len: int = 250
    num_classes: int = 5
    lae_variant: str = "resnet"
    embed_dim: int = 512
    gae_depth: int = 2
    gae_heads: int = 4
    gae_mlp_dim: int = 512
    patch_time: int = 6
    patch_freq: int = 4
    max_pyramid_scale: int = 16


class PaperBTFNet(nn.Module):
    def __init__(self, cfg: BTFNetConfig = BTFNetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.bridge = BridgeTF(max_pyramid_scale=cfg.max_pyramid_scale)

        if cfg.lae_variant.lower() == "convnext":
            self.lae_t = LAEConvNeXt(in_ch=cfg.in_channels, out_dim=cfg.embed_dim)
            self.lae_f = LAEConvNeXt(in_ch=cfg.in_channels * 2, out_dim=cfg.embed_dim)
        else:
            self.lae_t = LAEResNet1D(in_ch=cfg.in_channels, out_dim=cfg.embed_dim)
            self.lae_f = LAEResNet1D(in_ch=cfg.in_channels * 2, out_dim=cfg.embed_dim)

        self.gae_t = GAETransformer(
            in_ch=cfg.in_channels,
            seq_len=cfg.input_len,
            patch_size=cfg.patch_time,
            embed_dim=cfg.embed_dim,
            depth=cfg.gae_depth,
            num_heads=cfg.gae_heads,
            mlp_dim=cfg.gae_mlp_dim,
        )
        freq_len = cfg.input_len // 2 + 1
        self.gae_f = GAETransformer(
            in_ch=cfg.in_channels * 2,
            seq_len=freq_len,
            patch_size=cfg.patch_freq,
            embed_dim=cfg.embed_dim,
            depth=cfg.gae_depth,
            num_heads=cfg.gae_heads,
            mlp_dim=cfg.gae_mlp_dim,
        )

        self.mmf_t = MultiViewMutualFusion(dim=cfg.embed_dim)
        self.mmf_f = MultiViewMutualFusion(dim=cfg.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(cfg.embed_dim, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,L) or (B,C,L), got {tuple(x.shape)}")
        x = F.adaptive_avg_pool1d(x, self.cfg.input_len)

        x_t, x_f = self.bridge(x)
        l_t = self.lae_t(x_t)
        l_f = self.lae_f(x_f)
        g_t = self.gae_t(x_t)
        g_f = self.gae_f(x_f)

        f_t = self.mmf_t(l_t, g_t)
        f_f = self.mmf_f(l_f, g_f)

        fused = torch.cat([f_t, f_f], dim=1)
        logits = self.classifier(fused)
        if return_probs:
            return logits, torch.softmax(logits, dim=1)
        return logits


class BTFNetCrossEntropyLoss(nn.Module):
    

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        return self.ce(logits, targets.long())


def build_btfnet_model(cfg: BTFNetConfig | None = None) -> PaperBTFNet:
    return PaperBTFNet(cfg if cfg is not None else BTFNetConfig())


def build_btfnet_loss(label_smoothing: float = 0.0) -> BTFNetCrossEntropyLoss:
    return BTFNetCrossEntropyLoss(label_smoothing=label_smoothing)


def create_training_components(num_classes: int, labels=None):
    config = BTFNetConfig(in_channels=12, num_classes=num_classes, input_len=250, lae_variant="resnet")
    return PaperBTFNet(config), BTFNetCrossEntropyLoss()


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return x


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits = forward_logits(model, x)
    return logits, criterion(logits, y)

TRAINING_CONFIG = {"batch_size": 128, "learning_rate": 1e-4, "optimizer": "Adam", "scheduler": {"type": "step", "step_size": 10, "gamma": 0.95}}
