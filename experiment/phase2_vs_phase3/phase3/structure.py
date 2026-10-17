"""
phase3 侧组件：RMSNorm + RoPE + 标准 MHA（带 RoPE）+ FFN
（遗留的 PositionalEncoding / encoder-decoder 类已去掉，本实验不用）
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """
    均方根归一化：只做 RMS 尺度归一化，不做均值中心化、无 bias，比 LayerNorm 更快。
    x * rsqrt(mean(x²) + eps) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    RoPE 预计算：θ_i = pos / base^(2i/dim)
    返回半重复布局 [cos, cos]（配合 rotate_half 使用），形状 [end, dim]。
    """
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    对 q/k（[B, T, H, D] 布局，transpose 之前）施加旋转位置编码。
    cos/sin 形状 [T, D] → unsqueeze(unsqueeze_dim=1) → [T, 1, D]，对 H 维广播。
    """
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力 — Vaswani et al., Eq. 1
    Attention(Q,K,V) = softmax(QK^T / √d_k)V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)

    # 全 mask 行 softmax 会返回 NaN，替换为 0
    attention_weights = torch.where(torch.isnan(attention_weights),
                                    torch.zeros_like(attention_weights),
                                    attention_weights)
    return torch.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力 + RoPE（标准 MHA：4 头，每头各有自己的 K/V，本实验无 GQA）
    q/k 先过 RoPE 再进入注意力。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, cos=None, sin=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v)

        # RoPE：[B, T, H, D] 布局（还没 transpose 之前）旋转 Q 和 K
        if cos is not None:
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.dropout(x)
        return self.W_o(x), attn_weights


class PositionWiseFFN(nn.Module):
    """
    逐位置前馈网络 — Vaswani et al., Eq. 2
    d_model -> d_ffn -> d_model（在各位置独立应用）
    """
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, X):
        return self.net(X)
