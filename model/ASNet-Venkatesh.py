from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return self.norm(q + self.drop(h))


class FeatureTransformerEncoder(nn.Module):
    

    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x, x, x)
        h = self.ffn(x)
        return self.norm(x + self.drop(h))


class FeatureTransformerDecoder(nn.Module):
    

    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.agg_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        x = self.agg_attn(x, x, x)
        x = self.cross_attn(x, enc_out, enc_out)
        h = self.ffn(x)
        return self.norm(x + self.drop(h))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentiveTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x, x, x)
        h = self.ffn(x)
        return self.norm(x + self.drop(h))


class AttentiveTransformerDecoder(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.masked_self_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.masked_self_attn(x, x, x, attn_mask=mask)
        x = self.cross_attn(x, enc_out, enc_out)
        h = self.ffn(x)
        return self.norm(x + self.drop(h))


@dataclass
class ASNetConfig:
    in_channels: int = 1
    num_classes: int = 4
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    seq_len: int = 128
    attentive_decoder_layers: int = 8


class PaperASNet(nn.Module):
    

    def __init__(self, cfg: ASNetConfig = ASNetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.in_channels, cfg.d_model)
        self.input_bn = nn.BatchNorm1d(cfg.in_channels)
        self.ft_enc = FeatureTransformerEncoder(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
        self.ft_dec = FeatureTransformerDecoder(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=4096)
        self.at_enc = AttentiveTransformerEncoder(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout)
        self.at_dec_layers = nn.ModuleList(
            [AttentiveTransformerDecoder(cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.attentive_decoder_layers)]
        )
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B,L) or (B,C,L), got {tuple(x.shape)}")
        x = F.adaptive_avg_pool1d(x, self.cfg.seq_len)
        x = self.input_bn(x)
        x = x.transpose(1, 2).contiguous()
        x = self.input_proj(x)
        return x

    def forward(self, x: torch.Tensor, return_probs: bool = False):
        z = self._prepare_input(x)
        z_enc = self.ft_enc(z)
        z_ft = self.ft_dec(z, z_enc)
        a = self.pos_enc(z_ft)
        a_enc = self.at_enc(a)
        a_dec = a
        for dec in self.at_dec_layers:
            a_dec = dec(a_dec, a_enc)

        pooled = a_dec.mean(dim=1)
        logits = self.cls_head(pooled)
        if return_probs:
            probs = torch.softmax(logits, dim=1)
            return logits, probs
        return logits


class ASNetCrossEntropyLoss(nn.Module):
    

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 2:
            targets = targets.argmax(dim=1)
        return self.ce(logits, targets.long())


def build_asnet_model(cfg: ASNetConfig | None = None) -> PaperASNet:
    return PaperASNet(cfg if cfg is not None else ASNetConfig())


def build_asnet_loss(label_smoothing: float = 0.0) -> ASNetCrossEntropyLoss:
    return ASNetCrossEntropyLoss(label_smoothing=label_smoothing)

TRAINING_CONFIG = {"batch_size": 64, "learning_rate": 0.001, "optimizer": "Adam", "scheduler": {"type": "step", "step_size": 10, "gamma": 0.8}}


def create_training_components(num_classes: int, labels=None):
    return PaperASNet(ASNetConfig(in_channels=12, num_classes=num_classes)), ASNetCrossEntropyLoss()


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return x


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits = forward_logits(model, x)
    return logits, criterion(logits, y)
