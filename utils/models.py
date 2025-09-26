# utils/models.py
# -*- coding: utf-8 -*-
import math, torch
import torch.nn as nn

# -------- Transformer --------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerWL(nn.Module):
    def __init__(self, in_dim=2, d_model=64, nhead=4, num_layers=2, ffn=128, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn,
            batch_first=True, dropout=dropout, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, seq, key_padding_mask=None):
        h = self.embed(seq)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)

# -------- RetNet --------
class DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=padding, bias=True)
    def forward(self, x):
        y = self.dw(x.permute(0,2,1).contiguous())
        return y.permute(0,2,1).contiguous()

class RetentionCore(nn.Module):
    def __init__(self, d_model: int, conv_kernel: int = 5, dropout: float = 0.0):
        super().__init__()
        self.decay_logit = nn.Parameter(torch.zeros(d_model))
        self.proj_out    = nn.Linear(d_model, d_model)
        self.dwconv      = DepthwiseConv1d(d_model, kernel_size=conv_kernel)
        self.drop        = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        B, T, D = x.shape
        a = torch.sigmoid(self.decay_logit).view(1, D).expand(B, D)  # [B,D]
        s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            xt = x[:, t, :]
            s  = a * s + (1.0 - a) * xt
            ys.append(self.proj_out(s))
        y_ret = torch.stack(ys, dim=1)   # [B,T,D]
        y_loc = self.dwconv(x)           # [B,T,D]
        out = y_ret + y_loc
        return self.drop(out)

class RetNetBlock(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 2, dropout: float = 0.0, conv_kernel: int = 5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ret   = RetentionCore(d_model, conv_kernel=conv_kernel, dropout=dropout)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        hid = d_model * mlp_ratio
        self.mlp   = nn.Sequential(nn.Linear(d_model, hid), nn.GELU(), nn.Linear(hid, d_model))
    def forward(self, x):
        y = self.ret(self.norm1(x)); x = x + self.drop(y)
        y = self.mlp(self.norm2(x)); x = x + self.drop(y)
        return x

class RetNetWL(nn.Module):
    def __init__(self, in_dim=2, d_model=64, layers=4, mlp_ratio=2, dropout=0.0, conv_kernel=5):
        super().__init__()
        self.embed  = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([RetNetBlock(d_model, mlp_ratio, dropout, conv_kernel) for _ in range(layers)])
        self.head   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, seq, key_padding_mask=None):
        h = self.embed(seq)
        for blk in self.blocks:
            h = blk(h)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)

# -------- Mamba --------
class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError("Missing 'mamba-ssm'. pip install mamba-ssm causal-conv1d") from e
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model))
    def forward(self, x):
        y = self.mamba(self.norm1(x)); x = x + self.drop(y)
        y = self.mlp(self.norm2(x));   x = x + self.drop(y)
        return x

class MambaWL(nn.Module):
    def __init__(self, in_dim=2, d_model=64, layers=4, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, seq, key_padding_mask=None):
        h = self.embed(seq)
        for blk in self.blocks:
            h = blk(h)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)

# -------- RWKV (GRU fallback) --------
class RWKVBackbone(nn.Module):
    def __init__(self, d_model: int, layers: int, n_head: int = 4, dropout: float = 0.0):
        super().__init__()
        self.use_rwkv = False
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rwkv_stack = None
        try:
            from rwkv.model import RWKV  # may vary by version
            self.rwkv_stack = nn.ModuleList([RWKV(
                model_type="RWKV-4", n_layer=1, n_embd=d_model, n_head=n_head
            ) for _ in range(layers)])
            self.use_rwkv = True
        except Exception:
            self.use_rwkv = False
            self.rwkv_stack = None
        if not self.use_rwkv:
            self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=layers,
                              batch_first=True, bidirectional=False)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        h = self.norm(x)
        if self.use_rwkv and self.rwkv_stack is not None:
            for rw in self.rwkv_stack:
                y = rw(h)
                h = h + self.dropout(y)
        else:
            y, _ = self.gru(h); h = y
        return h

class RWKVWL(nn.Module):
    def __init__(self, in_dim=2, d_model=64, layers=4, n_head=4, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.backbone = RWKVBackbone(d_model=d_model, layers=layers, n_head=n_head, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
    def forward(self, seq, key_padding_mask=None):
        h = self.embed(seq)
        h = self.backbone(h)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)

# -------- Hyena --------
class _HyenaOpFallback(nn.Module):
    """Lightweight Hyena-style operator: Conv1d + GLU approximation.
       Ref: https://github.com/Suro-One/Hyena-Hierarchy"""
    def __init__(self, d_model: int, hidden_mult: int = 2, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model * hidden_mult * 2, kernel_size,
                               padding=pad, dilation=dilation, bias=True)
        self.glu = nn.GLU(dim=1)
        self.conv2 = nn.Conv1d(d_model * hidden_mult, d_model, 1, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # x: [B,T,D]
        h = self.norm(x)
        h = h.transpose(1, 2)                  # [B,D,T]
        h = self.conv1(h)                      # [B, 2*D*mult, T]
        h = self.glu(h)                        # [B, D*mult, T]
        h = self.conv2(h)                      # [B, D, T]
        h = h.transpose(1, 2).contiguous()     # [B,T,D]
        return self.drop(h)

class HyenaBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        # Use local fallback directly; do not attempt the official implementation
        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = _HyenaOpFallback(d_model, dropout=dropout)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.GELU(), nn.Linear(d_model*2, d_model)
        )

    def forward(self, x):
        y = self.hyena(self.norm1(x)); x = x + self.drop(y)
        y = self.mlp(self.norm2(x));   x = x + self.drop(y)
        return x

class HyenaWL(nn.Module):
    def __init__(self, in_dim=2, d_model=64, layers=4, dropout=0.0):
        super().__init__()
        self.embed  = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([HyenaBlock(d_model, dropout) for _ in range(layers)])
        self.head   = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, seq, key_padding_mask=None):
        h = self.embed(seq)
        for blk in self.blocks:
            h = blk(h)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)

# -------- MEGA --------
class MEGAWL(nn.Module):
    """
    MEGA for Water Level Estimation
    Ref: https://github.com/facebookresearch/mega
    """
    def __init__(self, in_dim=2, d_model=128, nhead=4, num_layers=4, dropout=0.1, ffn=256):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, seq, key_padding_mask=None):
        h = self.embed(seq)  # [B,T,D]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # [B,T,D]
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)
        
# -------- HGRN (local) --------
class HGRNCell(nn.Module):
    """
    Gated unit with per-feature diagonal recurrence (local implementation):
    s_t = sigma(g_z) ⊙ s_{t-1} + (1 - sigma(g_z)) ⊙ tanh(W x_t + sigma(g_r) ⊙ (u ⊙ s_{t-1}))
    where u is a learnable diagonal (per-channel) recurrent weight.
    Ref: https://github.com/OpenNLPLab/HGRN
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.u = nn.Parameter(torch.zeros(d_model))       # diagonal recurrent weight
        self.in_proj = nn.Linear(d_model, 3*d_model)      # [candidate, g_z, g_r]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B,T,D]
        x = self.norm(x)
        B, T, D = x.shape
        s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        u = self.u.view(1, D).expand(B, D)
        ys = []
        for t in range(T):
            xt = x[:, t, :]                              # [B,D]
            cand, g_z, g_r = self.in_proj(xt).chunk(3, dim=-1)
            z = torch.sigmoid(g_z)
            r = torch.sigmoid(g_r)
            h_tilde = torch.tanh(cand + r * (u * s))
            s = z * s + (1.0 - z) * h_tilde
            ys.append(s)
        y = torch.stack(ys, dim=1)                       # [B,T,D]
        return self.drop(y)

class DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=padding, bias=True)
    def forward(self, x):  # x: [B,T,D]
        return self.dw(x.permute(0,2,1)).permute(0,2,1).contiguous()

class HGRNBlock(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 2, dropout: float = 0.0, conv_kernel: int = 5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.core  = HGRNCell(d_model, dropout=dropout)
        self.loc   = DepthwiseConv1d(d_model, kernel_size=conv_kernel) if conv_kernel and conv_kernel > 1 else nn.Identity()
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        hid = d_model * mlp_ratio
        self.mlp   = nn.Sequential(nn.Linear(d_model, hid), nn.GELU(), nn.Linear(hid, d_model))

    def forward(self, x):
        h = self.norm1(x)
        y = self.core(h) + self.loc(h)   # global memory + local convolution
        x = x + self.drop(y)
        y = self.mlp(self.norm2(x))
        x = x + self.drop(y)
        return x

class HGRNWL(nn.Module):
    def __init__(self, in_dim=2, d_model=64, layers=4, mlp_ratio=2, dropout=0.0, conv_kernel=5):
        super().__init__()
        self.embed  = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([HGRNBlock(d_model, mlp_ratio, dropout, conv_kernel) for _ in range(layers)])
        self.head   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, seq, key_padding_mask=None):
        h = self.embed(seq)                       # [B,T,D]
        for blk in self.blocks:
            h = blk(h)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            h = (h * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        else:
            h = h.mean(dim=1)
        return self.head(h).squeeze(-1)

