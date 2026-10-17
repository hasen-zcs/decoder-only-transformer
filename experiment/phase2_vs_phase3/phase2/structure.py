"""
phase2 侧组件：标准 MHA（无 RoPE）+ FFN + AddNorm（LayerNorm 版）
位置信息由 model.py 的可学习位置表提供。
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


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
    多头注意力 — Vaswani et al., Sec 3.2.2
    标准 MHA：4 头，每头各有自己的 K/V。无位置编码。
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

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

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


class AddNorm(nn.Module):
    """
    残差连接 + 层归一化 — Vaswani et al., Sec 5.4
    output = LayerNorm(x + Dropout(Sublayer(x)))
    Post-LN 架构（论文原版）。
    """
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, residual, sublayer_output):
        return self.ln(residual + self.dropout(sublayer_output))
