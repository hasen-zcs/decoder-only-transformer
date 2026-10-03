"""
Transformer 模型实现（Vaswani et al., 2017）

Base: d_model=512, heads=8, layers=6, FFN=2048, ~93M params
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from structure import *


class GPTConfig:
    vocab_size = 4242
    block_size = 128
    n_embd = 128
    b_head = 4
    n_layer = 4


# --- 编码器 & 解码器层

class EncoderLayer(nn.Module):
    """
    编码器层：Self-Attention + FFN，各带残差连接。
    """
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, X, mask=None):
        attn_output, _ = self.self_attn(X, X, X, mask)
        X = self.add_norm1(X, attn_output)
        ffn_output = self.ffn(X)
        X = self.add_norm2(X, ffn_output)
        return X


class DecoderLayer(nn.Module):
    """
    解码器层：Masked Self-Attn + Cross-Attn + FFN，各带残差连接。
    """
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, X, enc_output, tgt_mask=None, src_mask=None):
        attn_output, _ = self.self_attn(X, X, X, tgt_mask)
        X = self.add_norm1(X, attn_output)
        cross_attn_output, _ = self.cross_attn(X, enc_output, enc_output, src_mask)
        X = self.add_norm2(X, cross_attn_output)
        ffn_output = self.ffn(X)
        X = self.add_norm3(X, ffn_output)
        return X


# --- Transformer 模型

class Transformer(nn.Module):
    """
    完整 Transformer 模型 — Vaswani et al., Sec 3.1

    三个入口：forward() 训练, encode() 编码, decode() 解码（支持逐步生成）
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ffn=2048, dropout=0.1,
                 max_len=5000, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ffn, dropout)
                          for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ffn, dropout)
                          for _ in range(num_decoder_layers)])
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 初始化 — Vaswani et al., Sec 3.2.2"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -- 掩码生成

    def generate_square_subsequent_mask(self, sz):
        """
        因果掩码 — 下三角矩阵，True 表示可见。
        """
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask

    def make_src_mask(self, src):
        """padding 掩码，返回 [batch, 1, 1, src_len] 支持广播。"""
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        print("src_mask shape: ",src_mask.shape)
        return src_mask

    def make_tgt_mask(self, tgt):
        """目标掩码 = 因果掩码 & padding 掩码。"""
        tgt_len = tgt.size(1)
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return padding_mask & subsequent_mask

    # -- 前向传播

    def encode(self, src):
        src_padding_mask = self.make_src_mask(src)
        src_mask = src_padding_mask & src_padding_mask.transpose(-2, -1)
        # print("src_mask shape", src_mask.shape)
        src_embed = self.src_embed(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)

        enc_output = src_embed
        for layer in self.encoder:
            enc_output = layer(enc_output, src_mask)
        return enc_output, src_padding_mask

    def decode(self, tgt, encoder_output, src_mask):
        tgt_mask = self.make_tgt_mask(tgt)
        # print("tgt_mask shape:", tgt_mask.shape)
        tgt_embed = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)

        dec_output = tgt_embed
        for layer in self.decoder:
            dec_output = layer(dec_output, encoder_output, tgt_mask, src_mask)
        return dec_output

    def forward(self, src, tgt):
        encoder_output, src_mask = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output, src_mask)
        output = self.linear(decoder_output)
        return output


# --- 快速验证
if __name__ == "__main__":
    print("构建 Transformer Base 模型（~65M 参数）...")
    model = Transformer(
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        max_len=200,
        pad_idx=0
    )

    # 验证 forward（训练模式）
    print("\n[1] forward()")
    src = torch.randint(0, 32000, (32, 50))
    tgt = torch.randint(0, 32000, (32, 40))
    output = model(src, tgt)
    print(f"    输入: src {src.shape}, tgt {tgt.shape}")
    print(f"    输出: {output.shape}  ← 应为 [32, 40, 32000]")

    # 验证 encode + decode（推理逐步生成）
    print("\n[2] encode() + decode()")
    src = torch.randint(0, 32000, (4, 30))
    enc, src_mask = model.encode(src)
    print(f"    encode → enc {enc.shape}, mask {src_mask.shape}")

    tgt = torch.tensor([[2], [2], [2], [2]])
    for step in range(5):
        dec = model.decode(tgt, enc, src_mask)
        next_token = dec[:, -1:].argmax(dim=-1)
        tgt = torch.cat([tgt, next_token], dim=1)
    print(f"    decode 5 steps → tgt {tgt.shape}  ← 应为 [4, 6]")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.1f}M")
    print("所有测试通过")
