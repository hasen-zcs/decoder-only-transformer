from model.model import GPT, GPTConfig
from model.Tokenizer import MyTokenizer
import torch
from torch import nn
from Dataset import MyDataset
from model.Config import GPTConfig

# prompt = "从前"
# 实例化配置，模型，分词器，优化器
config = GPTConfig()
tokenizer = MyTokenizer(config.VOCAB_PATH)
gpt = GPT(config).to(config.device)


gpt.load_state_dict(
    torch.load("checkpoints/model.pth", 
               map_location=config.device)
               )


prompts = ["从前", "宋霸出刀", "她打开门", "女儿"]
for p in prompts:
    idx = tokenizer.encode(p).to(config.device).unsqueeze(0)
    out = gpt.generate(idx=idx, max_new_tokens=2000)
    text = tokenizer.decode(out[0].cpu().tolist())
    print(f"[提示]{p}\n{text}\n{'='*40}")
    with open("output/生成结果.txt", "a", encoding="utf-8") as f:
        f.write(f"[提示]{p}\n{text}\n{'='*40}")











# idx = tokenizer.encode(prompt).to(config.device)




# # [T] -> [1, T]
# idx = idx.unsqueeze(0)

# generated = gpt.generate(
#     idx,
#     max_new_tokens=10000
# )

# tokens = generated[0].detach().cpu().tolist()
# text = tokenizer.decode(tokens)
# # text = tokenizer.decode(
# #     generated[0]
# # )

# print(text)