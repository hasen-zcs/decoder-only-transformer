import math
import sys
import torch
from pathlib import Path

# —— 每次跑改这一行 ——
PHASE = "phase2"    # "phase2" 或 "phase3"（两套架构不能共用一个模型实例）
CKPT  = "final"     # "final" → checkpoints/model_final.pth   "latest" → checkpoints/step_latest.pt
# -----------------------

EXP = Path(__file__).resolve().parent
sys.path.append(str(EXP / PHASE))        # 该 phase 的 model.py / Config.py
sys.path.append(str(EXP.parents[1]))     # 项目根（Dataset.py、data/）

from model import GPT
from Config import GPTConfig
from Dataset import MyDataset

BLOCK = 128     # 两侧都以 128 训练
N = 100         # 统一评测协议：100 个 val batch

config = GPTConfig()
assert config.block_size == BLOCK, "Config.block_size 必须与训练时一致"
gpt = GPT(config).to(config.device).eval()

if CKPT == "final":
    ckpt_path = EXP / PHASE / "checkpoints" / "model_final.pth"
    sd = torch.load(ckpt_path, map_location=config.device, weights_only=True)
    step_str = "final"
else:
    ckpt_path = EXP / PHASE / "checkpoints" / "step_latest.pt"
    state = torch.load(ckpt_path, map_location=config.device, weights_only=True)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    step_str = state.get("step", "?")

gpt.load_state_dict(sd)

ds = MyDataset(str(EXP.parents[1] / "data" / "val.bin"), config.block_size, config.batch_size)
losses = []
with torch.no_grad():
    for _ in range(N):
        x, y = ds.get_batch(config.device)
        _, loss = gpt(x, y)
        losses.append(loss.item())

val = sum(losses) / len(losses)
print(f"{PHASE} | {ckpt_path.name} (step {step_str})")
print(f"val loss = {val:.4f}   PPL = {math.exp(val):.2f}   (100 batches, val.bin, block={BLOCK})")
