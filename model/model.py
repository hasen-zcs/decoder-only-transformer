"""
Transformer 模型实现（Vaswani et al., 2017）

Base: d_model=512, heads=8, layers=6, FFN=2048, ~93M params
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from structure import *
import json


class GPTConfig:
    vocab_size = 4242
    block_size = 128
    n_embd = 128
    ffn_dim = 256
    b_head = 4
    n_layer = 4
    dropout = 0.1

    VOCAB_PATH = "data/vocab.json"


class GPTEmbedding(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )
        # 可学习位置编码
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

    def forward(self, x):
        B, T = x.shape

        token_emb = self.token_embedding(x)
        position = torch.arange(
            T,
            device=x.device
        )

        position_emb = self.position_embedding(position)

        return token_emb + position_emb


class GPTLayer(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            config.n_embd, 
            config.b_head,
            config.dropout
            )
        
        self.ffn = PositionWiseFFN(
            config.n_embd,
            2*config.n_embd,
            config.dropout
            )
        
        self.add_norm1 = AddNorm(
            config.n_embd,
            config.dropout
            )
        
        self.add_norm2 = AddNorm(
            config.n_embd,
            config.dropout
            )

    def forward(self, x):
        # 生成mask
        T = x.size(1)
        mask = torch.tril(
            torch.ones(
                T, T,
                device=x.device,
                dtype=torch.bool
            )
        )

        att_out, _ = self.self_attn(x, x, x, mask=mask)

        x = self.add_norm1(x, att_out)
        x = self.add_norm2(x, self.ffn(x))
        return (x)

class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()

        self.embedding = GPTEmbedding(config)
        self.layers = nn.ModuleList([
            GPTLayer(config)
            for _ in range(config.n_layer)
        ])

        self.norm = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size
        )

    def forward(self, x, targets=None):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        if targets is not None:
            B, T, V = logits.shape
            logits = B
            loss = F.cross_entropy(logits, targets)
        return logits
    
    

if __name__ == "__main__":
    config = GPTConfig
    seq = "你好世界！"

    # 导入词表
    with open(config.VOCAB_PATH, "r", encoding="utf-8") as f:
        stoi = json.load(f)

    itos = {int(i):ch for ch, i in stoi.items()}
    # 把词仍进词表
    seq = [stoi[ch] for ch in seq]
    # print(seq)
    seq = torch.tensor(seq, dtype=torch.long)
    seq = seq.unsqueeze(0)

    gpt = GPT(config)

    logits = gpt(seq)

    print("logits shape:", logits.shape)
    # 词嵌入(输入必须是long类型)
    # embedder = GPTEmbedding(config)
    # x = embedder(seq)
    # print("embed shape", x.shape)
    # layer = GPTLayer(config)
    # out = layer(x)
    # print("x shape", x.shape)
    # print("out shape", out.shape)



