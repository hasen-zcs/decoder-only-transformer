import math
import torch
from model.model import GPT
from model.Config import GPTConfig
from Dataset import MyDataset

# —— 每次跑改这两行 ——
CKPT = "checkpoints/phase4_final/step_latest.pt"
BLOCK = 512        # 必须等于该 checkpoint 训练时的 block_size

config = GPTConfig()
config.block_size = BLOCK
gpt = GPT(config).to(config.device).eval()
state = torch.load(CKPT, map_location=config.device)
sd = state["model_state_dict"] if "model_state_dict" in state else state
gpt.load_state_dict(sd)

ds = MyDataset("data/val.bin", config.block_size, config.batch_size)
N = 100
losses = []
with torch.no_grad():
    for _ in range(N):
        x, y = ds.get_batch(config.device)
        _, loss = gpt(x, y)
        losses.append(loss.item())
val = sum(losses) / len(losses)
print(f"{CKPT}\nval loss = {val:.4f}   PPL = {math.exp(val):.2f}   (100 batches, block={BLOCK})")