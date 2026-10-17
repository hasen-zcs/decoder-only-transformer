# 阶段二 vs 阶段三：架构消融实验

> 自包含实验：本文件夹里的代码独立运行，**与根目录 `model/` 无关**（那里已经改成 GQA+256，别混）。
> 运行位置：算力服务器，从项目根目录执行。

## 回答的问题

在**其余全部相同**的条件下，Post-LN + LayerNorm + 可学习位置表（阶段二）和 Pre-LN + RMSNorm + RoPE（阶段三）哪个架构更好？

## 单变量

**只动架构**（`phase2/` 与 `phase3/` 之间唯一不同的文件是 `model.py` / `structure.py`）。其余严格对齐：

| 项 | 值 |
|---|---|
| 数据 | 新数据集（9 本，vocab 6027；train 37,824,234 / val 4,719,841 token）|
| 规模 | 8 层 × n_embd 128，FFN = 2×n_embd = 512，4 头标准 MHA（**无 GQA**）|
| token | 10,000 步 × 64 batch × 128 block = 81.9M（≈ 新训练集 2.17 epoch）|
| 调度 | warmup 200（0→1e-3）+ cosine → 1e-4，clip 1.0，AdamW（wd 默认 0.01）|
| 其他 | dropout 0.1，tied lm_head，seed 42（torch + numpy 双播种，两边数据窗口序列完全一致）|

参数量（smoke.py 会核对）：

| 侧 | 参数量 | 差别的来源 |
|---|---|---|
| phase2 | **1,847,936** | 位置表 128×128 = 16,384 + LayerNorm 的 bias 2,176（每层 2×LN 各多 128 bias ×8 层 + 末尾 LN 128）|
| phase3 | **1,829,376** | RoPE 0 参数、RMSNorm 无 bias |

Δ = 18,560（≈1.0%），规模基本相同 → 对比公平。

## 目录

```
phase2/    Post-LN 架构（Config / tokenizer / structure / model / train / smoke / generate，自包含）
phase3/    Pre-LN+RMSNorm+RoPE 架构（同上）
eval_ppl.py  统一评测协议（100 batch val；两个架构不能共用一个模型，改 PHASE 跑两次）
plot_losses.py  把两侧 loss 曲线画进同一张图 → loss_curve.png（train/val 双子图，对数纵轴）
```

## 执行步骤（项目根目录，PowerShell）

```powershell
$PY = "D:\Users\17743\anaconda3\envs\minimind\python.exe"
```

### 0. 前提
- 服务器 `data/meta.json` 显示 `vocab_size: 6027`（新数据）
- `phase2/checkpoints/step_latest.pt`、`phase3/checkpoints/step_latest.pt` **都不存在**（新文件夹本来就没有；以后重跑要先删，否则是断点续训不是从零）

### 1. Smoke（1 分钟，验证两套模型接线正确）
```powershell
& $PY experiment/phase2_vs_phase3/phase2/smoke.py
& $PY experiment/phase2_vs_phase3/phase3/smoke.py
```
两边都应打印：`params` 与期望一致 + `argmax==x ≈ 1.0` + `smoke PASS`。
init loss 期望 **90~110**（tied 自尖峰指纹，见 `docx/03`）——loss 单独不作判据，argmax 才是真探针。
不 PASS → 停，先查接线再训练。

### 2. 训练（一个跑完再跑下一个，各 1 万步）
```powershell
& $PY experiment/phase2_vs_phase3/phase2/train.py *> experiment/phase2_vs_phase3/phase2/logs/train.log
& $PY experiment/phase2_vs_phase3/phase3/train.py  *> experiment/phase2_vs_phase3/phase3/logs/train.log
```
- 中途断了直接重跑同一条命令（自动从 `step_latest.pt` 续；会重跑最后一个已存 step 一次，无害）
- 日志每 100 步一行（step/train/val/lr/grad_norm）；两边都会从 ~100 起步、前几百步陡降（清自尖峰，正常）
- 每个 train.py 同步把每 100 步的数字写进各自 `logs/training_log.csv`（续训时自动去重不断线），**训练结束自动跑 `plot_losses.py`** 生成合并图 `loss_curve.png`；想随时刷新：`& $PY experiment/phase2_vs_phase3/plot_losses.py`

### 3. 评测（`eval_ppl.py` 顶部改 PHASE，跑两次）
```powershell
& $PY experiment/phase2_vs_phase3/eval_ppl.py   # PHASE="phase2"
# 把 PHASE 改成 "phase3" 再跑一次
& $PY experiment/phase2_vs_phase3/eval_ppl.py
```
输出：val loss + PPL（100 batch × 64 × 128，val.bin，block 128）——**最终数字以它为准**，不靠控制台每 100 步的 20-batch 采样。
- 曲线图：两侧都训完后再跑一次 `& $PY experiment/phase2_vs_phase3/plot_losses.py` → `loss_curve.png`（train/val 双子图，对数纵轴，英文标签避免服务器字体缺中文字形）

### 4. 采样盲评
```powershell
& $PY experiment/phase2_vs_phase3/phase2/generate.py
& $PY experiment/phase2_vs_phase3/phase3/generate.py
```
- 各 4 个 prompt（吕树 / 韩立 / 克莱恩 / 白小纯，均训练集里的人名）× 2000 字符，temperature 0.8 + top_k 50
- 结果追加到各自 `生成结果.txt`；盲评：不知道哪段是哪个模型，先判好坏再对答案

### 5. 记录 + 提交
- 填下面「结果表」+ `docx/README.md` 总表的阶段二/三行，然后 git commit（代码 + logs；`.pt` 不提交）

## 结果表（跑完填）

| 指标 | phase2（Post-LN） | phase3（Pre-LN+RoPE） |
|---|---|---|
| params | 1,847,936 | 1,829,376 |
| train loss @1000 / 5000 / 10000 | | |
| val loss @1000 / 5000 / 10000 | | |
| val loss @final（eval_ppl 100b） | | |
| PPL @final | | |
| 样本盲评 | | |
| 备注（grad_norm 顶 1.0？val 回升？其他） | | |

## 判读标准

- **主判据 = 最终 PPL**（token 对齐的 81.9M，最终对最终；中间曲线只作观察）
- phase3 小幅更好（PPL 低 1~3% 量级）→ 与文献一致：Pre-LN 训练更稳，规模越大越受益
- **几乎没差别也是结论**：~1.8M 参数、81.9M token 这个尺度上，架构选择不是主要矛盾，数据量/参数量才是——这本身就是一个消融发现
- phase2 反超 → 先别下结论，把两份 log 完整留着查（phase3 是否 warmup 不够 / val 是否回升）

## 纪律

- 本实验 GQA **关**（两侧都是标准 MHA）——GQA 是独立变量轴，留阶段五
- 训练进行中不改 Config；要改就另开文件夹
- checkpoint 各存各的 `checkpoints/`（每 1000 步 `step_latest.pt` + 结束 `model_final.pth`），互不干扰
