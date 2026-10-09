from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding1D(nn.Module):
    

    def __init__(self, in_ch: int, out_ch: int, dconv_kernel: int = 3, dconv_stride: int = 2) -> None:
        super().__init__()
        self.dconv = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size=dconv_kernel,
            stride=dconv_stride,
            padding=dconv_kernel // 2,
            groups=in_ch,
            bias=False,
        )
        self.pconv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=dconv_stride, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.skip(x) + self.pconv(self.dconv(x)))


class FGCA(nn.Module):
    

    def __init__(self, dim: int, window: int = 5) -> None:
        super().__init__()
        self.dim = dim
        self.window = window
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.k = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.t_bias = nn.Parameter(torch.randn(dim, window) * 0.02)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        q = self.q(x).transpose(1, 2)
        k = self.k(x)
        v = self.v(x)
        k_w = F.unfold(k.unsqueeze(-1), kernel_size=(self.window, 1), padding=(self.window // 2, 0)).view(b, c, self.window, l)
        v_w = F.unfold(v.unsqueeze(-1), kernel_size=(self.window, 1), padding=(self.window // 2, 0)).view(b, c, self.window, l)
        k_w = k_w.permute(0, 3, 1, 2)
        v_w = v_w.permute(0, 3, 1, 2)
        q_e = q.unsqueeze(-1)
        s = (q_e * k_w).sum(dim=2) / (self.dim ** 0.5)
        qb = torch.matmul(q, self.t_bias)
        att = torch.sigmoid(s + qb).unsqueeze(2)
        out = torch.max(att * v_w, dim=3).values
        out = out.transpose(1, 2).contiguous()
        return self.proj(out)


class GSA(nn.Module):
    

    def __init__(self, dim: int, n_tokens: int = 5) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.pre = nn.Sequential(nn.Conv1d(dim, dim, 1, bias=False), nn.GELU(), nn.BatchNorm1d(dim))
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Conv1d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        xg = self.pre(x)
        xg = F.adaptive_avg_pool1d(xg, self.n_tokens).transpose(1, 2)
        q = self.q(xg)
        k = self.k(xg)
        v = self.v(xg)
        att = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (c ** 0.5), dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).contiguous()
        out = F.interpolate(out, size=l, mode="linear", align_corners=False)
        return self.proj(out)


class IAA(nn.Module):
    

    def __init__(self, dim: int, window: int = 5, n_tokens: int = 5) -> None:
        super().__init__()
        self.fgca = FGCA(dim=dim, window=window)
        self.gsa = GSA(dim=dim, n_tokens=n_tokens)
        self.proj_f = nn.Conv1d(dim, dim, 1, bias=False)
        self.proj_g = nn.Conv1d(dim, dim, 1, bias=False)
        self.mix = nn.Conv1d(dim * 2, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.proj_f(self.fgca(x))
        z2 = self.proj_g(self.gsa(x))
        return self.mix(torch.cat([z1, z2], dim=1))


class FFN1D(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv1d(dim, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class VAEForDDE(nn.Module):
    

    def __init__(self, in_dim: int = 90, hidden_h: int = 20, hidden_l: int = 20) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
        self.lin_h = nn.Linear(in_dim, hidden_h)
        self.lin_l = nn.Linear(in_dim, hidden_l)
        self.dec = nn.Linear(hidden_h + hidden_l, in_dim)

    def encode(self, xw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv(xw.unsqueeze(1)).flatten(1)
        hh = self.lin_h(F.adaptive_avg_pool1d(x1.unsqueeze(1), self.lin_h.in_features).squeeze(1))
        hl = self.lin_l(xw)
        return hh, hl

    def decode(self, hh: torch.Tensor, hl: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([hh, hl], dim=1))

    def forward(self, xw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hh, hl = self.encode(xw)
        yw = self.decode(hh, hl)
        return hh, hl, yw


class DDE(nn.Module):
    

    def __init__(self, latent_dim: int = 40, n_centers: int = 4) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, latent_dim) * 0.1)
        self.n_centers = n_centers

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(h, self.centers).clamp_min(1e-8)
        inv = 1.0 / d
        u = inv / inv.sum(dim=1, keepdim=True)
        return u


class KnowledgeExtractor(nn.Module):
    

    def __init__(self, in_ch: int, in_len: int, p_dim: int = 90, d_dim: int = 50) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 16, 7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(32),
        )
        self.to_w = nn.Linear(16 * 32, p_dim)
        self.to_d = nn.Linear(p_dim, d_dim)
        self.in_len = in_len

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        xw = self.to_w(h)
        xd = self.to_d(xw)
        return xw, xd


class CFA(nn.Module):
    

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_t = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_t = nn.Conv1d(dim, dim, 1, bias=False)
        self.v_t = nn.Conv1d(dim, dim, 1, bias=False)
        self.q_k = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Conv1d(dim * 2, dim, 1, bias=False)

    def forward(self, xt: torch.Tensor, xk: torch.Tensor) -> torch.Tensor:
        b, c, l = xt.shape
        qt = self.q_t(xt).transpose(1, 2)
        kt = self.k_t(xt).transpose(1, 2)
        vt = self.v_t(xt).transpose(1, 2)

        qk = self.q_k(xk).unsqueeze(1)
        a_self = torch.softmax(torch.matmul(qt, kt.transpose(1, 2)) / (c ** 0.5), dim=-1)
        a_cross = 1.0 - torch.softmax(torch.matmul(qk, kt.transpose(1, 2)) / (c ** 0.5), dim=-1)
        a_cross = a_cross.repeat(1, l, 1)

        o_self = torch.matmul(a_self, vt)
        o_cross = torch.matmul(a_cross, vt)
        out = torch.cat([o_cross, o_self], dim=-1).transpose(1, 2).contiguous()
        return self.proj(out)


@dataclass
class MSAFormConfig:
    in_channels: int = 1
    input_len: int = 250
    num_classes: int = 4
    dim: int = 50
    ffn_hidden: int = 256
    n1: int = 2
    n2: int = 1
    fgca_window: int = 5
    gsa_tokens: int = 5
    p_dim: int = 90
    n_centers: int = 4


class PaperMSAForm(nn.Module):
    def __init__(self, cfg: MSAFormConfig = MSAFormConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbedding1D(cfg.in_channels, cfg.dim, dconv_kernel=3, dconv_stride=2)

        self.iaa_blocks = nn.ModuleList([IAA(cfg.dim, cfg.fgca_window, cfg.gsa_tokens) for _ in range(cfg.n1)])
        self.ffn1_blocks = nn.ModuleList([FFN1D(cfg.dim, cfg.ffn_hidden) for _ in range(cfg.n1)])
        self.ln1 = nn.ModuleList([nn.BatchNorm1d(cfg.dim) for _ in range(cfg.n1)])
        self.ln1_ffn = nn.ModuleList([nn.BatchNorm1d(cfg.dim) for _ in range(cfg.n1)])

        self.ke = KnowledgeExtractor(cfg.in_channels, cfg.input_len, p_dim=cfg.p_dim, d_dim=cfg.dim)
        self.vae = VAEForDDE(in_dim=cfg.p_dim, hidden_h=20, hidden_l=20)
        self.dde = DDE(latent_dim=40, n_centers=cfg.n_centers)

        self.cfa_blocks = nn.ModuleList([CFA(cfg.dim) for _ in range(cfg.n2)])
        self.ffn2_blocks = nn.ModuleList([FFN1D(cfg.dim, cfg.ffn_hidden) for _ in range(cfg.n2)])
        self.ln2 = nn.ModuleList([nn.BatchNorm1d(cfg.dim) for _ in range(cfg.n2)])
        self.ln2_ffn = nn.ModuleList([nn.BatchNorm1d(cfg.dim) for _ in range(cfg.n2)])

        self.head = nn.Linear(cfg.dim + cfg.n_centers, cfg.num_classes)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected (B,L) or (B,C,L), got {tuple(x.shape)}")
        x = F.adaptive_avg_pool1d(x, self.cfg.input_len)
        xt = self.patch(x)
        for i in range(self.cfg.n1):
            h = xt + self.iaa_blocks[i](self.ln1[i](xt))
            xt = h + self.ffn1_blocks[i](self.ln1_ffn[i](h))
        xw, xk = self.ke(x)
        hh, hl, yw = self.vae(xw)
        u = self.dde(torch.cat([hh, hl], dim=1))
        h2 = xt
        for i in range(self.cfg.n2):
            h = h2 + self.cfa_blocks[i](self.ln2[i](h2), xk)
            h2 = h + self.ffn2_blocks[i](self.ln2_ffn[i](h))

        f2 = F.adaptive_avg_pool1d(h2, 1).squeeze(-1)
        logits = self.head(torch.cat([u, f2], dim=1))
        if return_aux:
            return logits, {"xw": xw, "yw": yw, "hh": hh, "hl": hl, "u": u}
        return logits


class MSAFormLoss(nn.Module):
    

    def __init__(self, eps: float = 1e-6, vae_weight: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.vae_weight = vae_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if targets.ndim == 2:
            t = targets.float()
        else:
            t = F.one_hot(targets.long(), num_classes=logits.size(1)).float()

        probs = torch.softmax(logits, dim=1).clamp(self.eps, 1.0 - self.eps)
        cls = -(t * torch.log(probs) + (1.0 - t) * (1.0 - torch.log(probs)))
        cls_loss = cls.mean()

        if aux is None:
            return cls_loss

        vae_loss = F.mse_loss(aux["yw"], aux["xw"])
        return cls_loss + self.vae_weight * vae_loss


def build_msaform_model(cfg: MSAFormConfig | None = None) -> PaperMSAForm:
    return PaperMSAForm(cfg if cfg is not None else MSAFormConfig())


def build_msaform_loss(eps: float = 1e-6, vae_weight: float = 0.1) -> MSAFormLoss:
    return MSAFormLoss(eps=eps, vae_weight=vae_weight)


def create_training_components(num_classes: int, labels=None):
    return PaperMSAForm(MSAFormConfig(in_channels=12, num_classes=num_classes)), MSAFormLoss(eps=1e-6, vae_weight=0.1)


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    return x


def forward_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(prepare_input(x))


def forward_and_loss(model, criterion, x: torch.Tensor, y: torch.Tensor):
    logits, aux = model(prepare_input(x), return_aux=True)
    return logits, criterion(logits, y, aux)

TRAINING_CONFIG = {"batch_size": 256, "learning_rate": 4e-4, "optimizer": "Adam", "scheduler": {"type": "linear", "start": 4e-4, "end": 4e-5}}
