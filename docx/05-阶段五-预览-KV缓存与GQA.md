# 阶段五（预览）：KV Cache + GQA

> 仅预览，细节到阶段五资料时再展开。

## 生成现状（为什么值得升级）

`GPT.generate` 没有 KV cache：生成 2000 个 token，每步都对**最近 512 个 token 完整重算一遍前向**：

- 总计算量 ≈ 2000 × 512 = **1,024,000 个"token-当量"**
- 其中约 99.75% 是在重复计算早已算过的历史

## 升级 1：KV Cache

- 每层缓存已算位置的 k/v；第 t 步只对**1 个新 token** 做前向（它的 q 去 attend 缓存的 k/v）
- 总计算量 ≈ 512（首次）+ 1999×1 ≈ **2,511 个 token-当量** → 加速约 **400 倍**
- 实现要点：
  - `past_key_values`（每层一个 (k, v) 元组的列表）传进 `GPT.forward` / `MultiHeadAttention.forward`
  - RoPE 位置偏移：新 token 的位置 = 缓存长度，cos/sin 要按**绝对位置**取（表已预计算到 block_size，天然支持）
  - 缓存保留最近 block_size 个位置（与现在的截断行为一致）
- 收益：2000 token 生成从分钟级到秒级，长文本（万字级）生成变得可行

## 升级 2：GQA（Grouped Query Attention）

- `n_kv_heads = 2`：q 仍 4 头，每 2 个 q 头共享一组 kv 头
- 代码改动约 10 行：k/v 投影 `128 → 64` 维；split 后 `repeat_kv(x, 2)`（函数已写好，`structure.py`）
- 效果：KV 内存减半（与 KV cache 叠加后更明显）、参数 −约 131k（1.60M → 1.47M）、val loss 预期 ±0.05 内
- 顺序：**先 KV cache，后 GQA** —— 没有 cache 时 GQA 收益甚微

## 可选方向（视阶段五进度）

- 容量扩展：8→12 层 或 n_embd 128→256（测容量上限，PPL 对比，注意 token 对齐）
- embedding 初始化 N(0,1) → N(0, 0.02)（消除初始自尖峰，开局曲线平滑）
- 分词方向：字符级 4242 词表是学习选择；真实项目会换 subword（BPE），词表变小、序列变短

## 阶段五验收预览

- 2000 token 生成耗时：无 cache vs 有 cache（应差 2~3 个数量级）
- GQA 开/关：PPL 差 ≤ 0.05，KV 内存减半
- 生成质量盲评不下降
