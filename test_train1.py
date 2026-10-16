import torch
from model.model import GPT
from model.Config import GPTConfig

torch.manual_seed(0)
config = GPTConfig()
gpt = GPT(config)
gpt.eval()
x = torch.randint(0, config.vocab_size, (4, 128))

# 1) 看"最大的那个 logit"是不是同一个 token（公共分量的指纹）

topi = gpt(x).argmax(dim=-1)
print("argmax==当前token 的比例:", round((topi == x).float().mean().item(), 4))

topi = gpt(x).argmax(dim=-1)                 # [4,128]
ids, counts = topi.unique(return_counts=True)
print("top1 token 分布(出现次数>100的):")
print(ids[counts > 100].tolist(), counts[counts > 100].tolist())
v_star = ids[counts > 100][0].item()

# 2) 逐层分解：残差流在 e_v* 方向上的分量有多大
es = gpt.lm_head.weight[v_star]
es = es / es.norm()
h = gpt.embedding(x)
print("proj@embedding:   ", round((h @ es).mean().item(), 2))
for i, layer in enumerate(gpt.layers):
    h = layer(h, cos=gpt.freqs_cos[:128], sin=gpt.freqs_sin[:128])
    print(f"proj@layer{i+1}后:   ", round((h @ es).mean().item(), 2))
hn = gpt.norm(h)
print("proj@final norm:  ", round((hn @ es).mean().item(), 2),
      "  ← 若假设成立应 ≈ 8")


