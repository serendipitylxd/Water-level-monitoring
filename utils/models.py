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

# -------- RWKV（GRU 回退） --------
class RWKVBackbone(nn.Module):
    def __init__(self, d_model: int, layers: int, n_head: int = 4, dropout: float = 0.0):
        super().__init__()
        self.use_rwkv = False
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rwkv_stack = None
        try:
            from rwkv.model import RWKV  
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
    """轻量 Hyena 风格算子：Conv1d + GLU 近似。    
       参考: https://github.com/Suro-One/Hyena-Hierarchy"""
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
    参考: https://github.com/facebookresearch/mega
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
    逐特征对角递归的门控单元（本地实现）：
    s_t = sigma(g_z) ⊙ s_{t-1} + (1 - sigma(g_z)) ⊙ tanh(Wx_t + sigma(g_r) ⊙ (u ⊙ s_{t-1}))
    其中 u 是可学习的对角（逐通道）递归权。
    参考: https://github.com/OpenNLPLab/HGRN
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.u = nn.Parameter(torch.zeros(d_model))       
        self.in_proj = nn.Linear(d_model, 3*d_model)      # [cand, g_z, g_r]
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
        y = self.core(h) + self.loc(h)   
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


# ================================================================
# Simple baselines for ablation study
# Category:
#   1) Linear baseline: Linear Regression / Ridge Regression
#   2) Classical ML: SVR / Random Forest / XGBoost
#   3) Lightweight NN: MLP / 1D-CNN
#
# These baselines are added for low-dimensional wall-profile inputs.
# The PyTorch models keep the same forward interface as the existing
# sequence models: forward(seq, key_padding_mask=None), where
# seq: [N, T, C], usually [batch_size, num_sections, 2].
# ================================================================


def _flatten_wall_profile(seq, key_padding_mask=None, num_sections: int = 200):
    """
    Convert wall-profile sequence to a fixed-length vector.

    Args:
        seq: Tensor with shape [N, T, C] or [T, C].
        key_padding_mask: Optional bool Tensor with shape [N, T].
                          True means invalid / padded section.
        num_sections: Fixed number of sections used by linear/MLP baselines.

    Returns:
        x: Tensor with shape [N, num_sections * C].

    Notes:
        - If T < num_sections, zero padding is applied.
        - If T > num_sections, the sequence is truncated.
        - Invalid sections indicated by key_padding_mask are set to zero.
    """
    if seq.dim() == 2:
        seq = seq.unsqueeze(0)

    if seq.dim() != 3:
        raise ValueError(f"Expected seq with shape [N, T, C] or [T, C], but got {tuple(seq.shape)}")

    N, T, C = seq.shape
    x = seq

    if key_padding_mask is not None:
        if key_padding_mask.dim() == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0)
        if key_padding_mask.shape[:2] != (N, T):
            raise ValueError(
                f"key_padding_mask should have shape [N, T] = [{N}, {T}], "
                f"but got {tuple(key_padding_mask.shape)}"
            )
        valid = (~key_padding_mask).to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        x = x * valid

    if T < num_sections:
        pad = torch.zeros(N, num_sections - T, C, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)
    elif T > num_sections:
        x = x[:, :num_sections, :]

    return x.reshape(N, num_sections * C)


class LinearRegressionWL(nn.Module):
    """
    Linear regression baseline for water-level estimation.

    This model directly maps the flattened wall-profile vector Z in R^(B*in_dim)
    to one scalar water-level value. It is intended as the simplest baseline for
    verifying whether a linear mapping is sufficient.
    """
    def __init__(self, in_dim: int = 2, num_sections: int = 200, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_sections = num_sections
        self.regressor = nn.Linear(num_sections * in_dim, 1, bias=bias)

    def forward(self, seq, key_padding_mask=None):
        x = _flatten_wall_profile(seq, key_padding_mask, self.num_sections)
        return self.regressor(x).squeeze(-1)


class RidgeRegressionWL(LinearRegressionWL):
    """
    Ridge regression baseline implemented as a linear model.

    In a PyTorch training loop, ridge regression can be realized by either:
      1) setting optimizer weight_decay, or
      2) adding model.regularization_loss() to the task loss.

    Example:
        loss = mse_loss(pred, target) + model.regularization_loss()
    """
    def __init__(self, in_dim: int = 2, num_sections: int = 200,
                 ridge_alpha: float = 1e-4, bias: bool = True):
        super().__init__(in_dim=in_dim, num_sections=num_sections, bias=bias)
        self.ridge_alpha = ridge_alpha

    def regularization_loss(self):
        weight = self.regressor.weight
        return self.ridge_alpha * torch.sum(weight ** 2)


class MLPWL(nn.Module):
    """
    Lightweight MLP baseline.

    The input wall-profile sequence is flattened into a compact vector and then
    regressed by fully connected layers. This baseline tests whether nonlinear
    regression without explicit attention/SSM modules is sufficient.
    """
    def __init__(self, in_dim: int = 2, num_sections: int = 200,
                 hidden_dims=(128, 64), dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_sections = num_sections

        dims = [num_sections * in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, seq, key_padding_mask=None):
        x = _flatten_wall_profile(seq, key_padding_mask, self.num_sections)
        return self.net(x).squeeze(-1)


class CNN1DWL(nn.Module):
    """
    Lightweight 1D-CNN baseline.

    This model applies Conv1d along the wall-belt section dimension. Compared
    with MLP, it explicitly models local spatial correlations between adjacent
    wall-profile sections while remaining much simpler than Transformer/Mamba.
    """
    def __init__(self, in_dim: int = 2, channels=(32, 64), kernel_size: int = 5,
                 dropout: float = 0.0, head_hidden: int = 64):
        super().__init__()
        padding = (kernel_size - 1) // 2
        conv_layers = []
        c_in = in_dim
        for c_out in channels:
            conv_layers.append(nn.Conv1d(c_in, c_out, kernel_size=kernel_size,
                                         padding=padding, bias=True))
            conv_layers.append(nn.BatchNorm1d(c_out))
            conv_layers.append(nn.GELU())
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            c_in = c_out
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Sequential(
            nn.Linear(c_in, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, seq, key_padding_mask=None):
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        if seq.dim() != 3:
            raise ValueError(f"Expected seq with shape [N, T, C] or [T, C], but got {tuple(seq.shape)}")

        x = seq
        if key_padding_mask is not None:
            if key_padding_mask.dim() == 1:
                key_padding_mask = key_padding_mask.unsqueeze(0)
            valid = (~key_padding_mask).to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            x = x * valid

        # [N, T, C] -> [N, C, T]
        h = self.conv(x.transpose(1, 2).contiguous())

        # Masked global average pooling along section dimension.
        if key_padding_mask is not None:
            valid = (~key_padding_mask).to(dtype=h.dtype, device=h.device).unsqueeze(1)  # [N,1,T]
            h = (h * valid).sum(dim=-1) / (valid.sum(dim=-1) + 1e-6)
        else:
            h = h.mean(dim=-1)

        return self.head(h).squeeze(-1)


# ---------------- Classical ML baselines: sklearn-style wrappers ----------------
class _SKLearnWallProfileRegressor:
    """
    Base wrapper for classical machine-learning regressors.

    These models are not nn.Module and should be trained with fit()/predict(),
    not with a PyTorch optimizer. They are useful for ablation tables requested
    by reviewers, including SVR, Random Forest, and XGBoost.
    """
    def __init__(self, estimator, in_dim: int = 2, num_sections: int = 200):
        self.estimator = estimator
        self.in_dim = in_dim
        self.num_sections = num_sections

    def _to_numpy_features(self, seq, key_padding_mask=None):
        if not torch.is_tensor(seq):
            seq = torch.as_tensor(seq, dtype=torch.float32)
        if key_padding_mask is not None and not torch.is_tensor(key_padding_mask):
            key_padding_mask = torch.as_tensor(key_padding_mask, dtype=torch.bool)
        with torch.no_grad():
            x = _flatten_wall_profile(seq, key_padding_mask, self.num_sections)
        return x.detach().cpu().numpy()

    def fit(self, seq, target, key_padding_mask=None):
        x = self._to_numpy_features(seq, key_padding_mask)
        if torch.is_tensor(target):
            y = target.detach().cpu().numpy()
        else:
            y = target
        self.estimator.fit(x, y)
        return self

    def predict(self, seq, key_padding_mask=None):
        x = self._to_numpy_features(seq, key_padding_mask)
        return self.estimator.predict(x)

    def score(self, seq, target, key_padding_mask=None):
        x = self._to_numpy_features(seq, key_padding_mask)
        if torch.is_tensor(target):
            y = target.detach().cpu().numpy()
        else:
            y = target
        return self.estimator.score(x, y)


class SVRWL(_SKLearnWallProfileRegressor):
    """Support Vector Regression baseline."""
    def __init__(self, in_dim: int = 2, num_sections: int = 200,
                 kernel: str = "rbf", C: float = 10.0, epsilon: float = 0.01, **kwargs):
        try:
            from sklearn.svm import SVR
        except Exception as e:
            raise ImportError("SVRWL requires scikit-learn. Install with: pip install scikit-learn") from e
        estimator = SVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)
        super().__init__(estimator=estimator, in_dim=in_dim, num_sections=num_sections)


class RandomForestWL(_SKLearnWallProfileRegressor):
    """Random Forest regression baseline."""
    def __init__(self, in_dim: int = 2, num_sections: int = 200,
                 n_estimators: int = 300, max_depth=None, random_state: int = 42,
                 n_jobs: int = -1, **kwargs):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except Exception as e:
            raise ImportError("RandomForestWL requires scikit-learn. Install with: pip install scikit-learn") from e
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
        super().__init__(estimator=estimator, in_dim=in_dim, num_sections=num_sections)


class XGBoostWL(_SKLearnWallProfileRegressor):
    """XGBoost regression baseline."""
    def __init__(self, in_dim: int = 2, num_sections: int = 200,
                 n_estimators: int = 500, max_depth: int = 4,
                 learning_rate: float = 0.03, subsample: float = 0.9,
                 colsample_bytree: float = 0.9, random_state: int = 42,
                 objective: str = "reg:squarederror", **kwargs):
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise ImportError("XGBoostWL requires xgboost. Install with: pip install xgboost") from e
        estimator = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective=objective,
            **kwargs
        )
        super().__init__(estimator=estimator, in_dim=in_dim, num_sections=num_sections)
