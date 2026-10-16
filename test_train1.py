from model.model import GPT
from model.Config import GPTConfig
import torch

config = GPTConfig()
gpt = GPT(config)

print("freqs_cos:", gpt.freqs_cos.shape)
# 期望 (128, 32) = block_size × head_dim

print("权重共享生效:", gpt.lm_head.weight is gpt.embedding.token_embedding.weight)
# 期望 True

print("state_dict 键:", list(gpt.state_dict().keys()))
# 应无 position_embedding / add_norm*；应有 ln1/ln2、freqs_cos、freqs_sin

x = torch.randint(0, config.vocab_size, (4, 128))
logits, loss = gpt(x, x.roll(-1, dims=1))
loss.backward()
print("随机初始化 loss:", loss.item())
# 期望 ≈ ln(4242) = 8.35（随机模型的水平）

total = sum(p.numel() for p in gpt.parameters())
print(f"总参数: {total:,}")
# 期望 ≈ 1,071,488，比阶段二(≈1,089,024)少约 1.7 万