"""
训练前接线检查（先跑本文件，再跑 train.py）：
1. params 必须精确等于期望值 → 架构接线正确
2. loss 90~110 + argmax==x ≈ 1.0 → tied 自尖峰指纹（见 docx/03）
"""
import sys
from pathlib import Path
import torch

BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE.parents[2]))   # 项目根（本目录模块优先命中）

from model import GPT
from Config import GPTConfig

EXPECTED_PARAMS = 1847936   # phase2 期望值（phase3 侧是 1829376）

config = GPTConfig()
gpt = GPT(config)
params = sum(p.numel() for p in gpt.parameters())
ok_params = params == EXPECTED_PARAMS
print(f"params = {params:,}   expected = {EXPECTED_PARAMS:,}   → {'OK' if ok_params else '✗ MISMATCH，查 model.py！'}")
assert ok_params, "参数量不符，架构接线有问题"

gpt = gpt.to(config.device)
x = torch.randint(0, config.vocab_size, (4, config.block_size), device=config.device)
logits, loss = gpt(x, x.roll(-1, dims=1))
hit = (logits.argmax(-1) == x).float().mean().item()
print(f"init loss = {loss.item():.2f}   (期望 90~110，自尖峰)")
print(f"argmax==x = {hit:.3f}   (期望 ≈1.0，真探针)")
assert 90 <= loss.item() <= 110 and hit > 0.99, "指纹破坏，forward 有问题"
print("smoke PASS")
