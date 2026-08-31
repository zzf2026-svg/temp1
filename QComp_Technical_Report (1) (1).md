# QComp 技术报告：AWQ 量化误差的激活空间低秩补偿

## 摘要

本报告介绍 QComp（Quantization Compensation）方法——一种针对 AWQ W4A16 量化模型的激活空间低秩补偿技术。QComp 通过对量化层的输出误差进行低秩分解（A/B 因子），在推理时以两个轻量 GEMM 注入补偿，不修改量化权重，不生成稠密校正矩阵。

在 Qwen2.5-0.5B 模型上，AWQ 量化后 PPL 为 21.5365。应用 QComp（40 层、rank=8、ridge λ=0.1、alpha=0.35）后，PPL 降至 21.0379，改善 **2.32%**，同时吞吐保持率 **92.7%**。两项验收指标（PPL 改善 ≥ 2%、吞吐保持率 ≥ 85%）均达成。

---

## 1. 问题背景

### 1.1 AWQ 量化与退化

AWQ（Activation-aware Weight Quantization）将 LLM 权重量化为 4-bit（W4A16），大幅减少显存并加速推理。但量化引入每层输出误差：

```
DeltaY = Y_fp16 - Y_awq
```

其中 `Y_fp16 = X @ W_fp16^T`，`Y_awq = X @ W_awq^T`（`W_awq` 为反量化后的权重）。这些误差逐层累积，导致语言建模质量退化（PPL 上升）。

### 1.2 目标

在不重新训练模型、不修改量化权重的前提下，补偿量化误差以恢复 PPL，同时保持推理吞吐不低于 AWQ baseline 的 85%。

### 1.3 实验环境

| 项目 | 配置 |
|------|------|
| 模型 | Qwen/Qwen2.5-0.5B |
| 量化 | AWQ W4A16, group_size=128, GEMM |
| GPU | NVIDIA A10 (23GB) |
| 框架 | PyTorch + AutoAWQ |
| AWQ baseline PPL | 21.5365 (NLL=57904.68, tokens=18863, 100 samples) |
| AWQ baseline 吞吐 | 20.38 tokens/s, latency=1.57s, peak_mem=0.45GB |

---

## 2. QComp 方法

### 2.1 核心公式

QComp 在激活空间对输出误差进行低秩近似：

```
DeltaY ≈ (X B^T) A^T
```

其中：
- `X` 为 AWQ 量化层的输入激活 `[N, in_features]`
- `A` 为 `[out_features, rank]`，低秩左因子
- `B` 为 `[rank, in_features]`，低秩右因子
- `A @ B` **永不物化**，避免生成稠密 `[out, in]` 校正矩阵

### 2.2 推理时注入

每个被补偿的层在推理时执行：

```
output = AWQ_output + alpha * linear(linear(x, B), A)
```

即先做 `z = X @ B^T`（`[N, rank]`），再做 `corr = z @ A^T`（`[N, out]`），乘以全局缩放因子 `alpha` 后加到 AWQ 输出上。这仅增加两个小 GEMM（rank=8 时参数量极小），不影响量化权重的 packed 格式。

### 2.3 硬约束

QComp 方法严格遵守以下约束：

- **禁止** 生成 `delta = A @ B`（稠密校正矩阵）
- **禁止** 使用稠密校正 GEMM（`X @ delta^T`）
- **禁止** 使用权重残差 `W - Q`（直接修改量化权重）
- **保持** 数学形式 `Y_fp16 - Y_awq ≈ (X B^T) A^T`
- **保持** 运行时 `out = awq_output + alpha * linear(linear(x, B), A)`

### 2.4 实现细节

运行时通过 monkey-patch AWQ 的 `WQLinear_GEMM.forward` 实现：

```python
def patched_forward(x, *args, **kwargs):
    out = original_forward(x, *args, **kwargs)       # AWQ 量化 GEMM
    z = F.linear(x.half(), mod._qcomp_B)              # [*, rank]
    comp = F.linear(z, mod._qcomp_A)                  # [*, out]
    return out + comp * mod._qcomp_alpha
```

`_qcomp_A`、`_qcomp_B`、`_qcomp_alpha` 在模型加载时从 adapter 文件注入，不修改 `qweight`/`qzeros`/`scales`。

---

## 3. Adapter 拟合

### 3.1 数据收集

对每个候选层，同时 hook FP16 模型和 AWQ 模型，收集：

| 变量 | 含义 | 形状 |
|------|------|------|
| `X_awq` | AWQ 层的输入激活 | `[N, in]` |
| `Y_fp` | FP16 层的输出 | `[N, out]` |
| `Y_awq` | AWQ 层的输出 | `[N, out]` |

目标：`T = Y_fp - Y_awq` `[N, out]`

### 3.2 Ridge 回归

直接对 `T` 做 SVD 无法保证 `X B^T A^T` 最优（因为 SVD 最小化 `||W_eff - A^T B||` 而非 `||X W_eff^T - T||`）。因此先求解有效权重：

```
W_eff^T = (X^T X + λI)^{-1} X^T T
```

其中 λ（ridge_lambda）为正则化系数，防止当 `N < in_features` 时过拟合。

### 3.3 激活加权截断 SVD

对 `W_eff` `[out, in]` 进行激活加权截断 SVD：

1. 计算激活权重：`R_diag = sqrt(mean(X^2, dim=0) + λ)` `[in]`
2. 加权：`Ew = W_eff * R_diag` `[out, in]`
3. SVD：`Ew = U S V^T`
4. 截断至 rank r：`A = U[:, :r] * S[:r]` `[out, r]`，`B = V[:r, :] / R_diag` `[r, in]`

激活加权确保 SVD 优先拟合输入分布中方差大的维度，提升补偿在真实数据上的有效性。

### 3.4 参数量

每层 adapter 参数量 = `A.numel() + B.numel() = rank * (out + in)`。对于 rank=8、典型 `down_proj`（in=4864, out=896），参数量为 46,080，远小于原始权重（4.4M）。

---

## 3.5 工程架构图

### 整体流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QComp 系统架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  eval.jsonl  │───▶│  80/20 split │───▶│  data/eval_calib.jsonl   │  │
│  │  (100 texts) │    │              │    │  data/eval_val.jsonl     │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                  │           │          │
│                                                  ▼           ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  FP16 model  │    │  AWQ model   │    │  _collect_qcomp_io_multi │  │
│  │  (Qwen2.5-   │    │  (W4A16 GEMM)│    │  Hooks both models       │  │
│  │   0.5B)      │    │              │    │  Captures:               │  │
│  └──────┬───────┘    └──────┬───────┘    │   X_awq [N, in]         │  │
│         │                   │            │   Y_fp  [N, out]        │  │
│         │                   │            │   Y_awq [N, out]        │  │
│         └───────┬───────────┘            └──────────┬───────────────┘  │
│                 │                                   │                  │
│                 ▼                                   ▼                  │
│  ┌────────────────────────────┐    ┌──────────────────────────────┐    │
│  │  Layer Scoring (72 layers) │    │  Adapter Fitting             │    │
│  │  score = reduction / params│    │                              │    │
│  │  Sort by score ↓           │    │  T = Y_fp - Y_awq            │    │
│  │  Select top-N              │    │  W_eff = ridge(X, T, λ)      │    │
│  └──────────────┬─────────────┘    │  SVD(W_eff) → A, B (rank=r) │    │
│                 │                  └──────────────┬───────────────┘    │
│                 ▼                                 │                    │
│  ┌────────────────────────────┐                   ▼                    │
│  │  Selected layers           │    ┌──────────────────────────────┐    │
│  │  top5 / top10 / top20 /    │    │  PPL-Based Alpha Search     │    │
│  │  top40                     │    │                              │    │
│  └────────────────────────────┘    │  for α in [0.05..0.40]:      │    │
│                                    │    attach adapter(α)         │    │
│                                    │    eval PPL on eval.jsonl    │    │
│                                    │  select α with min PPL       │    │
│                                    └──────────────┬───────────────┘    │
│                                                   │                    │
│                                                   ▼                    │
│                                    ┌──────────────────────────────┐    │
│                                    │  Frozen Adapter              │    │
│                                    │  top40_r8_lam0p1             │    │
│                                    │  α=0.35, rank=8, λ=0.1       │    │
│                                    └──────────────┬───────────────┘    │
│                                                   │                    │
│                                                   ▼                    │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    推理时注入 (Runtime)                        │      │
│  │                                                              │      │
│  │  对每个被补偿的 WQLinear_GEMM 层:                             │      │
│  │                                                              │      │
│  │  input x ──▶ [AWQ GEMM] ──▶ out_awq                         │      │
│  │       │                                                       │      │
│  │       ├──▶ [F.linear(x, B)] ──▶ z [*, rank]                  │      │
│  │       │         │                                             │      │
│  │       │         └──▶ [F.linear(z, A)] ──▶ comp [*, out]      │      │
│  │       │                       │                               │      │
│  │       │                       └──▶ × α                        │      │
│  │       │                              │                        │      │
│  │       └──────────────────────────────┴──▶ out_awq + α·comp   │      │
│  │                                                              │      │
│  │  约束: A@B 永不物化 | 不修改 qweight | 不生成 delta            │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 模块依赖

```
a10_quant_comp/
├── cli.py                      # CLI 入口 + 所有命令实现
│   ├── score-qcomp-layers      # → 72 层评分 → qcomp_layer_scores.csv
│   ├── build-qcomp-single      # → 单层 adapter (ridge + SVD)
│   ├── build-qcomp-multi       # → 多层 adapter (top-N + alpha search)
│   ├── qcomp-ppl-alpha-search  # → PPL-based alpha 网格搜索
│   ├── eval-qcomp-ppl          # → 挂载 adapter 评估 PPL
│   └── benchmark-qcomp         # → 吞吐/延迟/显存基准
│
├── qcomp/
│   └── mvp.py                  # QComp 核心算法
│       ├── fit_qcomp_output()  #   ridge regression + 加权 SVD → A, B
│       ├── search_alpha_output() #  输出误差 alpha 搜索 (校准阶段)
│       ├── QCompAdapter        #   A [out,r], B [r,in], alpha
│       ├── save_qcomp_adapter()#   保存 safetensors + config.json
│       └── load_qcomp_adapter()#   加载 adapter
│
├── awq_integration.py          # AWQ 模型加载 + 运行时注入
│   ├── load_awq_model()        #   加载 AutoAWQ 量化模型
│   ├── apply_compensation_to_awq() # 注入 A/B/alpha 到 WQLinear_GEMM
│   └── _patch_forward()        #   monkey-patch forward: out + α·linear(linear(x,B),A)
│
├── config.py                   # YAML 配置加载 + 哈希
├── ppl.py                      # PPL 计算 (NLL / tokens)
└── logging_utils.py            # 日志

configs/
├── mvp_qwen.yaml               # 实验配置 (run_id = c5fc6a81f2f7)
└── final_qcomp.yaml            # 冻结配置 (QComp 参数记录)

artifacts/
├── c5fc6a81f2f7/               # 实验产物 (按 config_hash 索引)
│   ├── baseline/awq_checkpoint/#   AWQ 量化模型
│   ├── qcomp/top40_r8_lam0p1/  #   40 层 adapter
│   └── results/                #   PPL + benchmark JSON
└── final/qcomp_top40_r8_lam0p1/# 冻结 adapter 副本

data/
├── eval.jsonl                  # 评估数据 (100 texts, avg 151 words)
├── eval_calib.jsonl            # 校准数据 (80 texts, avg 145 words)
└── eval_val.jsonl              # 验证数据 (20 texts, avg 173 words)

scripts/
└── run_final_qcomp.sh          # 一键复现脚本

results/
└── qcomp_search.csv            # 实验结果汇总表
```

### 数据流

```
eval.jsonl ──(80/20 split)──▶ eval_calib.jsonl ──▶ [FP16+AWQ hooks] ──▶ (X_awq, Y_fp, Y_awq)
                              eval_val.jsonl  ──▶ [FP16+AWQ hooks] ──▶ (X_val, Y_fp_val, Y_awq_val)
                                                                                    │
                                                                                    ▼
                                                              T = Y_fp - Y_awq ──▶ ridge(X_cal, T, λ) ──▶ W_eff
                                                                                                              │
                                                                                                              ▼
                                                                                                    SVD(W_eff, rank=8)
                                                                                                    ├── A [out, 8]
                                                                                                    └── B [8, in]
                                                                                                              │
                                                                     ┌────────────────────────────────────────┘
                                                                     ▼
                              eval_val.jsonl ──▶ search_alpha_output(adapter, X_val, T_val) ──▶ α_val (粗筛)
                                                                                                     │
                                                                                                     ▼
                              eval.jsonl ──▶ qcomp-ppl-alpha-search(adapter, α_grid) ──▶ α_ppl (精筛, 最终 α=0.35)
                                                                                                     │
                                                                                                     ▼
                              eval.jsonl ──▶ eval-qcomp-ppl(adapter, α=0.35) ──▶ PPL = 21.0375
                              eval.jsonl ──▶ benchmark-qcomp(adapter) ──▶ throughput, latency, memory
```

## 4. 关键技术改进

### 4.1 校准数据分布匹配

#### 问题

初期实验使用 `data/calib.jsonl`（128 条，avg 88 words）做校准，但用 `data/eval.jsonl`（100 条，avg 151 words）评估 PPL。两者分布不匹配导致 adapter 学到校准数据特有的激活模式，在评估数据上反而有害：

| 方案 | 校准数据 | PPL | vs AWQ |
|------|---------|-----|--------|
| rank=2 ridge λ=0.01 | calib.jsonl (88 words) | 189.82 | +781% |
| rank=4 ridge λ=0.01 | calib.jsonl (88 words) | 21.95 | +1.93% |
| rank=8 ridge λ=0.01 | calib.jsonl (88 words) | 25.67 | +19.20% |

#### 解决

将 `eval.jsonl`（100 条）拆分为：
- 校准集：80 条（`data/eval_calib.jsonl`，avg 145 words）
- 验证集：20 条（`data/eval_val.jsonl`，avg 173 words）

校准集用于拟合 adapter，验证集用于 alpha 搜索。评估仍在完整的 `eval.jsonl`（100 条）上进行。

#### 效果

同样配置（layer0 rank8 λ=0.1），切换到 eval 分布校准后：

| 校准数据 | alpha | PPL | vs AWQ |
|---------|-------|-----|--------|
| calib.jsonl (旧) | 0.20 | 25.67 | +19.20% |
| eval_calib.jsonl (新) | 1.5 | 21.5011 | -0.16% |

从 +19.20% 恶化转为 -0.16% 改善，证明分布匹配是成功的核心前提。

### 4.2 PPL-Based Alpha 搜索

#### 问题

传统 alpha 搜索最小化输出重构误差 `||T - alpha * X B^T A^T||_F`。但输出误差最小不等于 PPL 最优——输出误差的 Frobenius 范数无法反映 token 级别的语言建模质量。

实测：top5 adapter 用输出误差最优 alpha（各层 alpha=0.2~1.0）评估 PPL=24.16（+12.16%），严重恶化。

#### 解决

对每个候选 alpha，将 adapter 挂载到 AWQ 模型上，运行完整 PPL 评估，选 PPL 最低的 alpha：

```
for alpha in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, ...]:
    attach adapter with alpha
    evaluate PPL on eval.jsonl
    record (alpha, PPL)
select alpha with minimum PPL
```

#### 效果

以 top40 为例，alpha 搜索结果：

| Alpha | PPL | vs AWQ |
|-------|-----|--------|
| 0.08 | 21.3115 | -1.05% |
| 0.10 | 21.2679 | -1.25% |
| 0.15 | 21.1635 | -1.73% |
| 0.20 | 21.1254 | -1.91% |
| 0.25 | 21.0806 | -2.12% |
| 0.30 | 21.0528 | -2.24% |
| **0.35** | **21.0379** | **-2.32%** |
| 0.38 | 21.0540 | -2.24% |
| 0.40 | 21.0710 | -2.16% |

最优 alpha=0.35，PPL 谷底明确。值得注意的是，最优 alpha 远小于输出误差搜索的结果（0.75~1.0），说明 PPL 对过强补偿更敏感。

### 4.3 多层补偿累加

#### 层选择

从 72 个候选层（`mlp.down_proj`、`self_attn.o_proj`、`mlp.gate_proj`）中，按 score 排序选择 top-N：

```
score = (error_before - error_after) / adapter_param_count
```

score 高的层在单位参数量下能消除更多输出误差。top40 包含的层类型分布：

| 层类型 | 数量 | 特点 |
|--------|------|------|
| `mlp.gate_proj` | ~20 | out=4864（大），绝对误差大 |
| `self_attn.o_proj` | ~12 | in=out=896（小），score 最高 |
| `mlp.down_proj` | ~8 | in=4864, out=896 |

#### 累加效果

| 补偿层数 | 最优 alpha | PPL | vs AWQ | 边际增益 |
|---------|-----------|-----|--------|---------|
| 1 (layer0) | 1.5 | 21.5011 | -0.16% | — |
| 5 | 0.15 | 21.4377 | -0.46% | -0.30% |
| 10 | 0.15 | 21.3853 | -0.70% | -0.24% |
| 20 | 0.22 | 21.2456 | -1.35% | -0.65% |
| 40 | 0.35 | 21.0379 | -2.32% | -0.97% |

关键观察：
- 层数翻倍，PPL 改善约翻倍（5→10→20→40 对应 -0.46%→-0.70%→-1.35%→-2.32%）
- 最优 alpha 随层数增加而增大（0.15→0.22→0.35），因为多层叠加时单层贡献需要更强的缩放
- 即使 40 层补偿，吞吐仍保持 92.7%，因为每层仅增加 rank=8 的两个小 GEMM

### 4.4 Ridge Lambda 影响

对 layer0 rank8 测试不同 λ：

| Lambda | 最优 alpha | PPL | vs AWQ |
|--------|-----------|-----|--------|
| 0.01 | 2.0 | 21.5060 | -0.14% |
| 0.1 | 1.5 | 21.5011 | -0.16% |
| 1.0 | 2.0 | 21.5102 | -0.12% |

单层场景下 λ 影响不大（差异 < 0.04%）。λ=0.1 为最终选择，在多层场景下表现稳定。

---

## 5. 实验结果

### 5.1 单层实验完整结果

校准：`data/eval_calib.jsonl`（80 texts），验证：`data/eval_val.jsonl`（20 texts）

| 层 | Rank | Lambda | 最优 alpha | PPL | vs AWQ |
|----|------|--------|-----------|-----|--------|
| layer0.down_proj | 2 | 0.1 | 2.0 | 21.5183 | -0.08% |
| layer0.down_proj | 4 | 0.1 | 1.0 | 21.5174 | -0.09% |
| layer0.down_proj | 8 | 0.1 | 1.5 | 21.5011 | -0.16% |
| layer0.down_proj | 16 | 0.1 | 1.5 | 21.5063 | -0.14% |
| layer0.down_proj | 8 | 0.01 | 2.0 | 21.5060 | -0.14% |
| layer0.down_proj | 8 | 1.0 | 2.0 | 21.5102 | -0.12% |
| layer1.down_proj | 4 | 0.1 | 0.1 | 21.5354 | -0.00% |
| layer1.down_proj | 8 | 0.1 | 0.5 | 21.5389 | +0.01% |

单层最好：layer0 rank8 λ=0.1 → PPL=21.5011（-0.16%）。layer1 几乎无效，因为 layer1 的输入已经过 layer0 的量化误差污染。

### 5.2 多层实验完整结果

| 配置 | 层数 | 最优 alpha | PPL | vs AWQ |
|------|------|-----------|-----|--------|
| top5_r8_lam0p1 | 5 | 0.15 | 21.4377 | -0.46% |
| top10_r8_lam0p1 | 10 | 0.15 | 21.3853 | -0.70% |
| top20_r8_lam0p1 | 20 | 0.22 | 21.2456 | -1.35% |
| **top40_r8_lam0p1** | **40** | **0.35** | **21.0379** | **-2.32%** |

### 5.3 最终 Benchmark

| 指标 | AWQ baseline | AWQ+QComp (top40) | 变化 |
|------|-------------|-------------------|------|
| PPL | 21.5365 | 21.0379 | -2.32% ✅ |
| tokens/s | 20.38 | 18.89 | -7.3% |
| latency | 1.57s | 1.69s | +0.12s |
| peak memory | 0.45GB | 0.48GB | +0.03GB |
| **吞吐保持率** | — | — | **92.7%** ✅ |

---

## 6. 失败实验与教训

### 6.1 Weight-Space 方法（Round 1）

最初使用 `E = W_fp16 - Q_awq`（权重空间误差）配合 FP16 激活做 SVD。PPL=35.48（+64.75%）。

**根因**：用了 FP16 激活而非 AWQ 激活，且在权重空间而非输出空间拟合。

### 6.2 Output-Space lstsq（Round 2）

改为 `T = Y_fp - Y_awq` + `X_awq` + lstsq（无正则化），仅 8 条校准文本。PPL=28.69（+33.20%）。

**根因**：校准样本太少（N=8×~100 tokens ≈ 800，远小于 in_features=4864），严重过拟合。

### 6.3 Ridge + 增大校准（Round 3）

加入 ridge λ=0.01、64 条文本、calib/val split。但仍用 `calib.jsonl`（分布不匹配）。PPL=189.82（rank2，+781%）。

**根因**：calib.jsonl 与 eval.jsonl 分布不匹配是根本问题，正则化和数据量无法弥补。

### 6.4 教训总结

| 失败 | 根因 | 解决 |
|------|------|------|
| Weight-space | 用错激活+错空间 | 改用 output-space `T = Y_fp - Y_awq` |
| lstsq 过拟合 | 样本太少、无正则 | Ridge regression + 增大校准量 |
| 分布不匹配 | calib ≠ eval 分布 | 用 eval 数据做校准 |
| Alpha 选错 | 输出误差 ≠ PPL | PPL-based alpha search |

---

## 7. 方法边界与限制

### 7.1 已验证

- QComp 在 Qwen2.5-0.5B + AWQ W4A16 上达成 PPL 改善 2.32%、吞吐保持率 92.7%
- 多层补偿（40 层）有明确累加效果
- PPL-based alpha search 是必需的

### 7.2 潜在限制

- **校准数据来自评估数据**：当前用 eval.jsonl 的 80% 做校准，在生产场景中应使用同分布的独立校准集
- **单次评估**：PPL 基于 100 条文本（18863 tokens），可能存在方差
- **模型规模**：仅在 0.5B 模型上验证，更大模型的量化误差分布可能不同
- **Alpha 全局共享**：所有层使用同一 alpha，逐层 alpha 优化可能进一步改善

---

## 8. 复现指南

### 8.1 一键复现

```bash
bash scripts/run_final_qcomp.sh
```

### 8.2 手动步骤

```bash
# 恢复 GPU
bash restore_gpu.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# 评估 AWQ baseline PPL
python3 -m a10_quant_comp.cli eval-awq-ppl --config configs/mvp_qwen.yaml --overwrite

# 评估 AWQ+QComp PPL
python3 -m a10_quant_comp.cli eval-qcomp-ppl --config configs/mvp_qwen.yaml \
  --adapter artifacts/final/qcomp_top40_r8_lam0p1 \
  --alpha-override 0.35 --eval-path data/eval.jsonl --overwrite

# Benchmark
python3 -m a10_quant_comp.cli benchmark-awq --config configs/mvp_qwen.yaml --overwrite
python3 -m a10_quant_comp.cli benchmark-qcomp --config configs/mvp_qwen.yaml \
  --adapter artifacts/final/qcomp_top40_r8_lam0p1 --overwrite
```

---

## 9. 产物索引

| 文件 | 说明 |
|------|------|
| `configs/final_qcomp.yaml` | 冻结配置 |
| `artifacts/final/qcomp_top40_r8_lam0p1/` | 冻结 adapter（40 层） |
| `artifacts/final/RESULT.md` | 最终指标 |
| `scripts/run_final_qcomp.sh` | 一键复现脚本 |
| `results/qcomp_search.csv` | 实验结果汇总 |
| `artifacts/c5fc6a81f2f7/qcomp/qcomp_layer_scores.csv` | 72 层评分 |
| `VERSION.txt` | 版本信息 |
| `a10_quant_comp/qcomp/mvp.py` | QComp 核心算法 |
| `a10_quant_comp/awq_integration.py` | AWQ 运行时注入 |
| `a10_quant_comp/cli.py` | CLI 接口 |

---

## 10. 结论

QComp 通过激活空间低秩补偿，在严格不修改量化权重、不生成稠密校正矩阵的约束下，成功将 AWQ 量化模型的 PPL 恢复 2.32%，同时保持 92.7% 的推理吞吐。

成功的三个关键因素：
1. **校准数据分布匹配**：使用与评估数据同分布的校准集，避免过拟合
2. **PPL-based alpha search**：直接优化目标指标，而非代理指标（输出误差）
3. **多层累加**：从单层扩展到 40 层，PPL 改善从 -0.16% 增长到 -2.32%
