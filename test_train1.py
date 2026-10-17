import time, torch
from model.model import GPT
from model.Config import GPTConfig
from model.Tokenizer import MyTokenizer

config = GPTConfig()
gpt = GPT(config).to(config.device).eval()
tok = MyTokenizer(config.VOCAB_PATH)
p = torch.tensor([[tok.stoi[c] for c in "从前"]], device=config.device)

# ① token 对齐：同种子各生成 150 个，找第一个不一致的位置
torch.manual_seed(123)
old = gpt.generate_nocache(p, 150)
torch.manual_seed(123)
new = gpt.generate(p, 150)
same = (old[0] == new[0]).int().tolist()
first_diff = next((i for i, s in enumerate(same) if not s), "150/150 全一致")
print("第一个不一致位置:", first_diff)
# 期望 ≥ 124（128 - prompt长2，窗口开始滑动前两边数学上完全相同）
# 124 之后分叉是预期（位置语义不同，见课前说明）；< 120 就分叉 = 有 bug

# ② 计时：500 token 加速比
torch.cuda.synchronize()
t0 = time.time(); gpt.generate_nocache(p, 500); torch.cuda.synchronize()
t_old = time.time() - t0
t0 = time.time(); gpt.generate(p, 500); torch.cuda.synchronize()
t_new = time.time() - t0
print(f"无 cache {t_old:.1f}s | 有 cache {t_new:.2f}s | 加速 {t_old/t_new:.0f}x")
# 期望几十倍（小 GPU 上 Python/kernel 开销会吃掉一部分理论值）