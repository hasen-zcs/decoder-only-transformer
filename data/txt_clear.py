from pathlib import Path
import json
import torch


BASE_DIR = Path(__file__).resolve().parent
print("base_dir:", BASE_DIR._str)
path = BASE_DIR / "novel.txt"
text = path.read_text(encoding="utf-8")

print("字符数：", len(text))
# print(text[:500])

text = text.replace("\r\n", "\n")
text = text.replace("\r", "\n")

lines = [line.strip() for line in text.split("\n") if line.strip()]
text = "\n".join(lines)

print("清理后字符数：", len(text))
# print(text[:500])

# 划分数据集验证集
split = int(len(text)*0.9)
train_text = text[:split]
val_text = text[split:]

print("训练集大小：", len(train_text))
print("验证集大小：", len(val_text))

# ------------构造简单字符表----------------
chars = sorted(set(text))
vocab_size = len(chars)
print("前100个字符:", chars[:100])
print("词表大小：", vocab_size)

# 建立字符于数字的爽映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 测试
s = "你好世界"
ids = [stoi[ch] for ch in s]
print("ids:", ids)
decoded = "".join(itos[i] for i in ids)
print(decoded)

train_ids = torch.tensor(
    [stoi[c] for c in train_text],
    dtype=torch.long
)
val_ids = torch.tensor(
    [stoi[c] for c in val_text],
    dtype=torch.long
)

print(train_ids.shape)
print(val_ids.shape)

block_size = 128
x = train_ids[:block_size]
y = train_ids[1:block_size+1]

print("x_chars:",[itos[i] for i in x[:20].tolist()])
print("y_chars:",[itos[i] for i in y[:20].tolist()])


# # 保存数据集
# torch.save(train_ids, BASE_DIR / "train.bin")
# # 保存验证集
# torch.save(train_ids, BASE_DIR / "train.bin")

# 保留原始二进制文件
import numpy as np
np.array(train_ids.numpy(), dtype=np.uint16).tofile(f"{BASE_DIR}/train.bin")
np.array(val_ids.numpy(), dtype=np.uint16).tofile(f"{BASE_DIR}/val.bin")

# 保存字表
with open(BASE_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)

meta = {
    "vocab_size": vocab_size,
    "encoding": "utf-8",
    "tokenizer": "char"
}
with open(BASE_DIR/"meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
