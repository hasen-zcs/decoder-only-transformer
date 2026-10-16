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




# 阶段三 将位置编码换成旋转位置编码
class GPTEmbedding(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

    def forward(self, x):
        token_emb = self.token_embedding(x)
        return token_emb


class GPTLayer(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.rms_norm1 = RMSNorm(config.n_embd)
        self.rms_norm2 = RMSNorm(config.n_embd)

        self.self_attn = GroupQueryAttention(
            config.n_embd, 
            config.b_head,
            config.n_kv_head,
            config.dropout
            )
        
        self.ffn = PositionWiseFFN(
            config.n_embd,
            2*config.n_embd,
            config.dropout
            )
        

    def forward(self, x, cos=None, sin=None):
        # 生成mask
        T = x.size(1)
        mask = torch.tril(
            torch.ones(
                T, T,
                device=x.device,
                dtype=torch.bool
            )
        )

        # 先RMSNorm再进入网络
        h = self.rms_norm1(x)

        att_out, _ = self.self_attn(h, h, h, mask=mask, cos=cos, sin=sin)

        x = x + att_out # 残差

        x = x + self.ffn(self.rms_norm2(x)) # FFN

        return x

class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config

        self.embedding = GPTEmbedding(config)
        self.layers = nn.ModuleList([
            GPTLayer(config)
            for _ in range(config.n_layer)
        ])

        # self.norm = nn.LayerNorm(config.n_embd)
        self.norm = RMSNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )
        # 权重共享：lm_head 直接服用 token 嵌入矩阵
        self.lm_head.weight = self.embedding.token_embedding.weight

        # RoPE预计算(阶段三)
        head_dim = config.n_embd//config.b_head
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, end=config.block_size)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)


    def forward(self, x, targets=None):
        x = self.embedding(x)

        T = x.size(1)
        # 取当前序列长度T对应的RoPE表（形状[T, head_dim]
        cos = self.freqs_cos[:T]
        sin = self.freqs_sin[:T]

        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin)

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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
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
            logits = logits[:, -1, :] / temperature
            # [B, vocab_size]

            # # 选择概率最大的 token
            # next_token = torch.argmax(logits, dim=-1, keepdim=True)
            # # [B, 1]

            # temperature + top-k
            v, _  = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接到原来的序列后面
            idx = torch.cat((idx, next_token), dim=1)

        return idx



    
    

if __name__ == "__main__":

    config = GPTConfig()
    gpt = GPT(config)
    a = gpt.layers[0].self_attn
    print(type(a).__name__)                            # GroupedQueryAttention
    print("W_k:", a.W_k.weight.shape, " W_v:", a.W_v.weight.shape)   # [64,128] [64,128]
    print("params:", sum(p.numel() for p in gpt.parameters()))        # 期望 1,697,280

    x = torch.randint(0, config.vocab_size, (2, 128))
    logits, loss = gpt(x, x.roll(-1, dims=1))
    print("loss:", round(loss.item(), 2))                          # 90~110
    print("argmax==x:", (logits.argmax(-1) == x).float().mean().item())  # ≈1.0



    # config = GPTConfig
    # seq = "你好世界！"

    # # 导入词表
    # with open(config.VOCAB_PATH, "r", encoding="utf-8") as f:
    #     stoi = json.load(f)

    # itos = {int(i):ch for ch, i in stoi.items()}
    # # 把词仍进词表
    # seq = [stoi[ch] for ch in seq]
    # # print(seq)
    # seq = torch.tensor(seq, dtype=torch.long)
    # idx = seq.unsqueeze(0)
    # x = idx[:, :-1]
    # target = idx[:, 1:]

    # gpt = GPT(config)

    # logits, loss = gpt(x, target)

    # print("logits shape:", logits.shape)
    # print("loss:", loss.item())

    # # 验证反向传播
    # loss.backward()
    # print(
    #     "lm_head grad:",
    #     gpt.lm_head.weight.grad is None
    # )

    # print(
    #     "embedding grad:",
    #     gpt.embedding.token_embedding.weight.grad is None
    # )



    # 词嵌入(输入必须是long类型)
    # embedder = GPTEmbedding(config)
    # x = embedder(seq)
    # print("embed shape", x.shape)
    # layer = GPTLayer(config)
    # out = layer(x)
    # print("x shape", x.shape)
    # print("out shape", out.shape)



