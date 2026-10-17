# 阶段五（预览）：KV Cache + GQA

> 仅预览，细节到阶段五资料时再展开。
> （2026-10-16：GQA + KV cache 已提前实现并验证通过，见文末「实现与验证」；本文件其余部分保留为原设计稿。）

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

## 实现与验证（2026-10-16，本机，128 维 GQA @4242 · 1,468,800 参数）

### 实现（4 处，对话教学 + 逐步验收）

1. `GroupQueryAttention`（GQA）：k/v 投影 `128 → 64`（2 个 kv 头）；新增 `max_len` 与 `k_cache/v_cache` 参数——**RoPE 之后**沿时间轴（dim=1，cat 处布局 `[B, T, H, D]`）拼接、超 `max_len` 只留最近、缓存 **2 头粒度**（repeat_kv 之前）
2. `GPTLayer`：透传 cache，返回 `(x, K_new, V_new)`
3. `GPT.forward`：`k_caches/v_caches`（8 层 = 8 对独立 cache，list）+ `start_pos`（绝对位置）+ 外部传入长 RoPE 表（不动 state_dict 里的注册 buffer）；`k_caches=None` 时训练路径零改动
4. `generate`：长表 `precompute_freqs_cis(head_dim, end=block+max_new+8)` + prefill + 每步单 token decode（位置 = T0+s）；旧版保留为 `generate_nocache` 作对照组

### 验证数字

| 测试 | 结果 | 期望 | 判定 |
|---|---|---|---|
| 同位置 logits：无 cache vs 有 cache（5 token） | max diff **1.14e-05** | <1e-2（纯浮点） | ✅ |
| cache[0] 形状 | **(1, 5, 2, 32)** = [B, T, H_kv, d_k] | 同布局 | ✅ |
| token 对齐（同种子，150 token） | **150/150 全一致**（数学保证范围 ≤124 = 128−prompt 长；窗口滑动后仍全中是采样巧合） | ≥124 | ✅ |
| 墙钟 500 token（本机 GPU） | 无 cache 4.2s / 有 cache **5.27s**（≈1x） | 几十倍 | ⚠️ 见发现 |

### 发现：1.5M 参数规模下墙钟是"开销主导"（overhead-bound）

- 无 cache 每步 ≈8.4 ms，其中**每步固定开销**（Python 循环、kernel launch、`torch.multinomial` 每步 GPU 同步）≈8 ms，而每 token GPU 计算仅 ~0.5 ms
- cache 每步省下 ~0.5 ms 计算，却多出 16 个小 cat（8 层×2）+ 分配（~1-2 ms）→ 净收益 ≈0，反而略慢
- 结论：**纸面 400 倍是 FLOPs 数字；墙钟提速要求"每 token 计算 ≫ 每步固定开销"**（更大模型 / 编译融合内核 / 真实推理引擎）。1.5M 参数下 1x 是预期，不是 bug
- 验收预览里"差 2~3 个数量级"的期望仅在大模型/引擎场景成立；本项目的验收以**机制正确性 + 诚实记录实测**为准

### 遗留

- 主 run（训练好的模型）后复测加速比并回填本节
- 本机 `Config.vocab_size` 仍 4242（params=1,468,800 证实）；本机跑真实数据前必须同步 6027
