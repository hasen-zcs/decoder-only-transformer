"""
训练完成后采样：4 个 prompt × 2000 字符（temperature 0.8 + top_k 50）
追加写入本目录 生成结果.txt，与 phase3 侧盲评。
"""
import sys
from pathlib import Path
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.append(str(ROOT))

from model import GPT
from Config import GPTConfig
from tokenizer import MyTokenizer

# 均来自训练集的书（《大王饶命》《凡人修仙传》《诡秘之主》《地球人实在太凶猛了》）
PROMPTS = ["吕树", "韩立", "克莱恩", "白小纯"]
MAX_NEW = 2000

config = GPTConfig()
gpt = GPT(config).to(config.device)

ckpt = BASE / "checkpoints" / "model_final.pth"
if not ckpt.exists():
    ckpt = BASE / "checkpoints" / "step_latest.pt"
state = torch.load(ckpt, map_location=config.device, weights_only=True)
sd = state["model_state_dict"] if "model_state_dict" in state else state
gpt.load_state_dict(sd)
gpt.eval()
print(f"loaded {ckpt.name}")

tok = MyTokenizer(str(ROOT / "data" / "vocab.json"))
out_path = BASE / "生成结果.txt"

with open(out_path, "a", encoding="utf-8") as f:
    f.write(f"\n===== phase2 · {ckpt.name} =====\n")
    for p in PROMPTS:
        ids = [tok.stoi[ch] for ch in p if ch in tok.stoi]   # 未知字符丢弃（正常不会发生）
        idx = torch.tensor([ids], dtype=torch.long, device=config.device)
        out = gpt.generate(idx, MAX_NEW, temperature=0.8, top_k=50)
        text = tok.decode(out[0].tolist())
        print(f"=== {p} ===\n{text}\n")
        f.write(f"--- {p} ---\n{text}\n")

print(f"written → {out_path}")
