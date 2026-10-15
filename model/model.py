"""
Transformer 模型实现（Vaswani et al., 2017）

Base: d_model=512, heads=8, layers=6, FFN=2048, ~93M params
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from .structure import *
import json
from .Config import GPTConfig





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
        self.config = config

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
            # cross_entropy是比较概率分布和目标之间的差值
            loss = F.cross_entropy(
                logits.view(B*T, V), 
                targets.view(B*T))
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
            根据已有的 token 序列，逐个生成新的 token。

            idx:
                [B, T]
                B = batch size
                T = 当前已有序列长度

            max_new_tokens:
                最多生成多少个新 token
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 如果当前序列超过 block_size，只保留最后 block_size 个 token
            idx_cond = idx[:, -self.config.block_size:]

            # 前向传播
            logits = self(idx_cond)

            # 只取最后一个位置的预测
            logits = logits[:, -1, :]
            # [B, vocab_size]

            # 选择概率最大的 token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            # [B, 1]

            # 拼接到原来的序列后面
            idx = torch.cat((idx, next_token), dim=1)

        return idx



    
    

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
    idx = seq.unsqueeze(0)
    x = idx[:, :-1]
    target = idx[:, 1:]

    gpt = GPT(config)

    logits, loss = gpt(x, target)

    print("logits shape:", logits.shape)
    print("loss:", loss.item())

    # 验证反向传播
    loss.backward()
    print(
        "lm_head grad:",
        gpt.lm_head.weight.grad is None
    )

    print(
        "embedding grad:",
        gpt.embedding.token_embedding.weight.grad is None
    )



    # 词嵌入(输入必须是long类型)
    # embedder = GPTEmbedding(config)
    # x = embedder(seq)
    # print("embed shape", x.shape)
    # layer = GPTLayer(config)
    # out = layer(x)
    # print("x shape", x.shape)
    # print("out shape", out.shape)



