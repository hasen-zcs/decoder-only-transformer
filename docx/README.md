# Decoder-Only Transformer · 阶段总结

> 字符级中文网文 GPT 学习项目。本文件夹按阶段记录全部工作：目标、改动、数据、踩坑与结论。
> 项目根目录：`D:\学校\个人日程\decoder-only-transformer`

## 项目概览

| 项 | 值 |
|---|---|
| 任务 | 字符级语言模型，下一字符预测 |
| 数据 | `data/novel.txt`（约 27.6 万行）→ `data/txt_clear.py` 清洗 → `data/train.bin` / `data/val.bin`（uint16） |
| 词表 | 字符级，4242 字（`data/vocab.json`） |
| 基础结构 | block 128，n_embd 128，ffn 256，4 头（head_dim 32），8 层，batch 64，dropout 0.1 |
| 规模 | 约 160 万参数（8 层共享版 = 1,600,896，已验证） |
| 环境 | conda env `minimind`，CUDA 训练，算力服务器 |
| 代码 | `model/model.py`（GPT）+ `model/structure.py`（attn/norm/RoPE）+ `train_mini.py` + `generate.py` + `Dataset.py` |

## 记录协议

每完成一阶段，用户报告效果，我负责：
1. 在对应阶段文件（01–05）里补「训练结果」小节：关键数字（train/val loss 曲线节点、PPL）、样本质量观察、与计划预期的对照、意外与处理。
2. 更新本文件「实验总表」的待填行。
3. 有新阶段/新发现时追加文件或勘误，并在「阶段索引」登记。
4. 提醒用户提交 git（文档 + 代码 + 日志）。

## 阶段索引

| 文件 | 阶段 | 一句话 | 状态 |
|---|---|---|---|
| [01-阶段一-首次训练与生成管线](01-阶段一-首次训练与生成管线.md) | 阶段一 | 首次 4 层训练 + 打通生成管线 | ✅ 完成 |
| [02-阶段二-训练技巧](02-阶段二-训练技巧.md) | 阶段二 | LR 调度 + 梯度裁剪 + 权重共享 + 8 层 | ⚠️ 共享配方未跑过，重跑中（服务器） |
| [03-阶段三-架构升级](03-阶段三-架构升级.md) | 阶段三 | Pre-LN + RMSNorm + RoPE | ⚠️ 1 万步训练中（服务器） |
| [04-阶段四-长上下文](04-阶段四-长上下文.md) | 阶段四 | block 128→512（token 对齐） | 📋 计划就绪 |
| [05-阶段五-预览-KV缓存与GQA](05-阶段五-预览-KV缓存与GQA.md) | 阶段五（预览） | KV cache + GQA | 📋 预览 |

## 实验总表

（随训练进展填写；`PPL = e^val_loss`）

| 项 | 阶段一 | 旧 8 层 untied run | 阶段二（重跑） | 阶段三 | 阶段四 |
|---|---|---|---|---|---|
| 架构 | 4L Post-LN+LN+位置表，untied | 8L Post-LN+LN+位置表，untied（判定见 02） | 8L Post-LN+LN+位置表，tied | 8L Pre-LN+RMSNorm+RoPE，tied | 同阶段三 + block 512 |
| 步数 / token | 1000 / 8.19M | 1000 / 8.19M | 10000 / 81.9M | 10000 / 81.9M | 2500 / 81.9M |
| 学习率 | 恒定 1e-3 | 恒定 1e-3 | warmup200+cosine→1e-4+clip1.0 | 同左 | 同左 |
| 种子 | — | — | 42 | 42 | 42 |
| 参数量 | ≈1.64M（估） | ≈2.17M（估） | ≈1,619,456（估） | **1,600,896（已验证）** | 1,600,896 |
| train loss@终 | 3.80 @999 | 1.53 @1000 | 待填 | 待填 | 待跑 |
| val loss@终 | — | 1.597 @1000 | 待填 | 待填（eval_ppl.py 100 batch） | 待跑 |
| PPL@终 | — | 4.94 | 待填 | 待填 | 待跑 |
| 4-prompt 样本质量 | A/B/C | — | 待填 | 待填 | 待跑 |

> 受控对：**阶段二 vs 阶段三**（仅架构不同）；**阶段三 vs 阶段四**（仅上下文长度不同）。
> "旧 8 层 untied run" 只能作历史参考，不能当阶段二基线（原因见 02 的发现）。

## 方法论沉淀（跨阶段）

1. **受控实验**：一次只动一个变量；步数与 token 量对齐（block 512 每步 token 是 128 的 4 倍，故 2500 步对齐 10000 步）；同种子；训练脚本升级后，旧基线必须用新脚本重跑（阶段二重跑就是这么来的）。
2. **Smoke test**：大训练前先验证随机初始化模型。本项目**共享（tied）模型**的初始 loss 指纹**不是** ln(4242)=8.35，而是 `loss ≈ 90~110` 且 `(argmax==x).mean() ≈ 1.0`（自尖峰机制，见 03）。指纹破坏 = 真 forward bug。
3. **日志即证据**：训练输出全部进 `logs/*.log`（每 100 步 val loss）；最终数字用统一协议 `eval_ppl.py`（100 batch）跑，不靠控制台肉眼。
4. **版本管理**：每阶段 git commit；旧架构代码留 `model/backup/`；checkpoint 用命名备份（`step_latest_1.pt`、`checkpoints/phaseX_final/`）；重要产物不要只留在工作区（阶段二 CSV 被删后靠 git 才找回）。
5. **断点续训**：`step_latest.pt` 存 model + optimizer，重跑同一命令即可续（已知小细节：会重跑最后一个已存 step 一次，无害）。

## 遗留清单

- [ ] 服务器阶段二重跑 / 阶段三 1 万步结果出来后，填总表
- [ ] 旧 checkpoint 属性确认（一行命令在 02）：`step_latest_1.pt` 的层数 / 是否 tied
- [ ] `.gitignore`：`*.pt`、`__pycache__/`、`output/`
- [ ] `train_mini.py` 续训 off-by-one：`start_step = checkpoint["step"] + 1`
- [ ] 此前交付的"阶段三"资料里 smoke 期望（8.35 / ≈40）已作废，以 03 的勘误为准
- [ ] 可选：embedding 初始化 N(0,1)→N(0, 0.02)（消除初始自尖峰，初始 loss 回到 ~8.4；做的话对应对比组要重跑）
- [ ] 阶段五资料：KV cache + GQA（见 05）
