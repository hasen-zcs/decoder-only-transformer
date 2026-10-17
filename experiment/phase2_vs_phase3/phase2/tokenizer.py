import json
import torch


class MyTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)

        self.itos = {
            int(i): ch
            for ch, i in self.stoi.items()
        }

    # 用@property装饰器将方法伪装成属性
    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        # 返回 torch.long，因为 nn.Embedding 的输入必须是 long
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, idx):
        return "".join(self.itos[i] for i in idx)
