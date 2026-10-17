"""
阶段二vs三消融实验 · phase2 侧
架构：Post-LN + LayerNorm + 可学习位置表 + tied lm_head（标准 MHA，无 GQA）
"""
import torch
from torch import nn
import torch.nn.functional as F
from structure import *
from Config import GPTConfig


class GPTEmbedding(nn.Module):
    """phase2：token 嵌入 + 可学习位置表"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

    def forward(self, x):
        B, T = x.shape

        token_emb = self.token_embedding(x)
        position = torch.arange(T, device=x.device)
        position_emb = self.position_embedding(position)

        return token_emb + position_emb


class GPTLayer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            config.n_embd,
            config.b_head,
            config.dropout
        )

        self.ffn = PositionWiseFFN(
            config.n_embd,
            2 * config.n_embd,
            config.dropout
        )

        self.add_norm1 = AddNorm(config.n_embd, config.dropout)
        self.add_norm2 = AddNorm(config.n_embd, config.dropout)

    def forward(self, x):
        # 因果mask
        T = x.size(1)
        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        att_out, _ = self.self_attn(x, x, x, mask=mask)

        # Post-LN：先残差相加，再归一化
        x = self.add_norm1(x, att_out)
        x = self.add_norm2(x, self.ffn(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
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
            config.vocab_size,
            bias=False
        )
        # 权重共享：lm_head 直接服用 token 嵌入矩阵
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, x, targets=None):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T)
            )
            return logits, loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        """
            idx: [B, T]，逐 token 采样生成。
            temperature + top_k 截断后 multinomial（阶段一打通的采样管线）。
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 超过 block_size 时只保留最后 block_size 个 token
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx
