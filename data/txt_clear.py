from pathlib import Path
import json
import re
import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "novel"
OUT_DIR = BASE_DIR

print("BASE_DIR:", BASE_DIR)
print("RAW_DIR :", RAW_DIR)

# 1.读单个文件
def read_text_robust(p:Path)->str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="ignore")

# ---------------- 2. 清洗单本书 ----------------
# 可按需增删的正则规则
AD_PATTERNS = [
    re.compile(r"https?://\S+"),
    re.compile(r"www\.\S+"),
    re.compile(r"^\s*(本章未完.*|请收藏.*|手机用户.*|笔趣阁.*)\s*$"),
    re.compile(r"^\s*第?\s*\d+\s*[章节回].{0,20}$"),   # 可选：去掉章节标题
]

# 清理文章
def clean_novel(text: str) -> str:
    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 全角空格 -> 半角
    text = text.replace("\u3000", " ")
    # 去 BOM
    text = text.lstrip("\ufeff")

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(p.search(line) for p in AD_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines)

# ---------------- 3. 遍历文件夹 ----------------
files = sorted(RAW_DIR.rglob("*.txt"))
print(f"发现 {len(files)} 个 txt 文件")

EOT = "<|endoftext|>"          # 书与书之间的分隔符
chunks = []

for i, f in enumerate(files):
    raw = read_text_robust(f)
    cleaned = clean_novel(raw)
    if len(cleaned) < 100:      # 太短的跳过
        print(f"[skip] {f.name} 清理后仅 {len(cleaned)} 字")
        continue
    chunks.append(cleaned)
    if (i + 1) % 20 == 0:
        print(f"已处理 {i+1}/{len(files)}")

# 用分隔符拼接所有书
full_text = (EOT + "\n").join(chunks)
print("清理后总字符数：", len(full_text))

# ---------------- 4. 划分 train / val ----------------
# 注意：按“整本书”划分比按字符切更合理，避免验证集里出现训练集同书片段
# 这里演示按书划分：前 90% 的书做训练，后 10% 做验证
n_books = len(chunks)
n_train = max(1, int(n_books * 0.9))
train_text = (EOT + "\n").join(chunks[:n_train])
val_text   = (EOT + "\n").join(chunks[n_train:])

print("训练集字符数：", len(train_text))
print("验证集字符数：", len(val_text))

# ---------------- 5. 构建字符表（只用 train，避免泄漏） ----------------
chars = sorted(set(train_text))
# 保证 EOT 一定在词表里
if EOT not in chars:
    # 特殊 token 用单字符占位不现实，这里改成逐字符统计即可；
    # 若想让 EOT 成为单个 id，需要改成 tokenizer 方案（见文末说明）
    pass

vocab_size = len(chars)
print("词表大小：", vocab_size)
assert vocab_size < 65535, "词表超过 uint16 上限，请改用 uint32 或 BPE"

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 处理 val 中可能出现的 train 未见字符
def encode(s: str, drop_unknown=True):
    if drop_unknown:
        return [stoi[c] for c in s if c in stoi]
    unk = len(chars)          # 给未知字符一个 id（可选）
    return [stoi.get(c, unk) for c in s]

train_ids = torch.tensor(encode(train_text), dtype=torch.long)
val_ids   = torch.tensor(encode(val_text),   dtype=torch.long)

print("train_ids:", train_ids.shape, "val_ids:", val_ids.shape)

# ---------------- 6. 保存 ----------------
train_ids.numpy().astype(np.uint16).tofile(OUT_DIR / "train.bin")
val_ids.numpy().astype(np.uint16).tofile(OUT_DIR / "val.bin")

with open(OUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(stoi, f, ensure_ascii=False, indent=2)

meta = {
    "vocab_size": vocab_size,
    "encoding": "utf-8",
    "tokenizer": "char",
    "eot": EOT,
    "n_books": n_books,
    "n_train_books": n_train,
    "train_chars": len(train_text),
    "val_chars": len(val_text),
}
with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("已保存到：", OUT_DIR)