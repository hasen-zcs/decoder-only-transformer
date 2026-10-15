from model.model import GPT, GPTConfig
from model.Tokenizer import MyTokenizer
import torch
from torch import nn

config = GPTConfig()
tokenizer = MyTokenizer(config.VOCAB_PATH)

gpt = GPT(config)

seq = "你好呀，我是新模型！"

idx = tokenizer.encode(seq)
print("idx:", idx)



idx = idx.unsqueeze(0)

x = idx[:, :-1]
target = idx[:, 1:]
# 前向传播
logits, loss = gpt(x, target)

# print("logit shape:", logits.shape)
# print("loss item:", loss.item())

old_weight = gpt.lm_head.weight.detach().clone()

optimizer = torch.optim.AdamW(
    gpt.parameters(),
    lr=1e-3
)
optimizer.zero_grad()
# 反向传播
loss.backward()
optimizer.step()

new_weight = gpt.lm_head.weight.detach().clone()

print("更新后参数是否相等：", torch.equal(old_weight, new_weight))


