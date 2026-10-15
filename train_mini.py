from model.model import GPT, GPTConfig
from model.Tokenizer import MyTokenizer
import torch
from torch import nn
from Dataset import MyDataset
from model.Config import GPTConfig
from pathlib import Path


# 实例化配置，模型，分词器，优化器
config = GPTConfig()
tokenizer = MyTokenizer(config.VOCAB_PATH)
gpt = GPT(config).to(config.device)
optimizer = torch.optim.AdamW(
    gpt.parameters(),
    lr=1e-3
)



# 导入数据
train_dataset = MyDataset(
    "data/train.bin", 
    config.block_size,
    config.batch_size
    )

val_dataset = MyDataset(
    "data/val.bin", 
    config.block_size,
    config.batch_size
    )

@torch.no_grad()
def evaluate(model, dataset, eval_steps, device="cuda"):
    model.eval()

    losses = []
    for _ in range(eval_steps):
        x, y = dataset.get_batch(device)

        logits, loss = gpt(x, y)

        losses.append(loss.item())
    model.train()

    return sum(losses) / len(losses)

def train_epochs(max_steps, start_step=None):
    if start_step is None:
        start_step = 0
    for step in range(start_step, max_steps):
        optimizer.zero_grad()
        # 取数据
        x, y = train_dataset.get_batch(config.device)
        # 前向传播
        logits, loss = gpt(x, y)
        # 反向传播
        loss.backward()
        # 更新优化器
        optimizer.step()

        if (step+1) % 100 == 0:

            val_loss = evaluate(
                gpt,
                val_dataset,
                eval_steps=20
            )
            print(
                f"step {step+1}, "
                f"train loss {loss:.4f}, "
                f"val loss {val_loss:.4f}"
            )

        if (step+1) % 1000 == 0:
            checkpoint = {
                        "step" : step,
                        "model_state_dict": gpt.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item()
                    }
            
            torch.save(
                        checkpoint,
                        "checkpoints/step_latest.pt"
                    )
        if step+1 == max_steps:
            torch.save(gpt.state_dict(), "checkpoints/model.pth")



# 测试是否可用
# print("train batch:", train_dataset.get_batch()[0].shape)

# checkpoints断点重续
p = Path("checkpoints/step_latest.pt")
if p.exists():
    checkpoint = torch.load(
        "checkpoints/step_latest.pt",
        map_location=config.device
    )
    gpt.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = checkpoint["step"]

if not p.exists():
    train_epochs(config.epochs, 0)

if start_step <= config.epochs:
    train_epochs(config.epochs, start_step)





