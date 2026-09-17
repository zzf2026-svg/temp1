# 大模型量化误差补偿算法测试报告

## 文档信息

| 项目 | 内容 |
|---|---|
| 文档名称 | 大模型量化误差补偿算法测试报告 |
| 被测算法 | QComp（Quantization Compensation）激活空间低秩量化误差补偿 |
| 被测模型 | Qwen/Qwen2.5-0.5B |
| 量化方案 | AWQ W4A16，group_size=128，GEMM |
| 最终配置 | Top-40 layers，rank=8，ridge λ=0.1，global alpha=0.35 |
| 测试平台 | NVIDIA A10 23GB，PyTorch + AutoAWQ |
| 报告生成日期 | 2026-09-17 |
| 测试结论 | **通过（PASS）** |

> 数据来源：本报告根据项目已有的 `QComp_Technical_Report.md` 和
> `qcomp_guidance_report.md` 中记录的测试数据整理生成。未提供的原始逐请求日志、
> 多次重复实验数据和 Git commit 信息不作补充推断。

---

## 1. 测试概述

### 1.1 测试背景

AWQ 将大语言模型权重量化为 W4A16，可以降低模型存储和推理资源开销，但量化会在各线性层引入输出误差。误差经过 Transformer 层逐层传播后，可能导致语言模型困惑度（Perplexity，PPL）上升。

QComp 的目标是在以下约束下补偿 AWQ 量化误差：

1. 不重新训练或修改 AWQ 量化权重；
2. 不生成稠密校正矩阵；
3. 通过低秩 A/B 因子在推理阶段注入轻量残差补偿；
4. 在恢复 PPL 的同时，控制额外推理开销。

### 1.2 测试目的

本次测试验证 QComp 是否满足以下验收目标：

| 验收指标 | 目标 |
|---|---:|
| 相对 AWQ baseline 的 PPL 改善 | ≥ 2% |
| 相对 AWQ baseline 的吞吐保持率 | ≥ 85% |

### 1.3 测试范围

本报告覆盖：

- AWQ baseline PPL 和推理性能；
- 单层 QComp 的 rank、ridge λ 和 alpha 测试；
- 多层 QComp 的层数扩展测试；
- PPL-based alpha 搜索；
- 最终 Top-40 配置的 PPL、吞吐、时延和显存测试；
- 算法结构约束检查；
- 失败实验与问题闭环分析。

本报告不覆盖：

- 更大参数规模模型；
- 除 PPL 外的下游任务准确率；
- 多 GPU、并发服务或真实线上流量；
- 独立于校准数据的严格盲测集；
- 多次重复测试的方差或置信区间。

---

## 2. 被测算法说明

### 2.1 输出误差建模

对一个 AWQ 量化线性层，收集：

- `X_awq`：AWQ 层输入激活，形状 `[N, in_features]`；
- `Y_fp`：FP16 对应层输出，形状 `[N, out_features]`；
- `Y_awq`：AWQ 对应层输出，形状 `[N, out_features]`。

补偿目标定义为：

```text
T = Y_fp - Y_awq
```

QComp 用低秩形式近似该输出误差：

```text
T ≈ (X B^T) A^T
```

其中：

- `A` 的形状为 `[out_features, rank]`；
- `B` 的形状为 `[rank, in_features]`；
- 最终配置的 `rank=8`。

### 2.2 Ridge 回归

先求解输入到目标输出误差的有效映射：

```text
W_eff^T = (X^T X + λI)^(-1) X^T T
```

其中 `λ` 为 ridge 正则化系数，用于缓解样本数量不足、特征相关性和矩阵病态导致的过拟合问题。

最终采用：

```text
ridge λ = 0.1
```

### 2.3 激活加权低秩分解

对 `W_eff` 进行激活加权截断 SVD：

```text
R_diag = sqrt(mean(X^2, dim=0) + λ)
Ew     = W_eff * R_diag
Ew     = U S V^T

A = U[:, :r] * S[:r]
B = V[:r, :] / R_diag
```

### 2.4 推理时补偿

运行时输出为：

```text
output = AWQ_output + alpha * linear(linear(x, B), A)
```

最终采用全局：

```text
alpha = 0.35
```

### 2.5 结构约束

测试所依据的最终实现满足以下约束：

- 不物化 `delta = A @ B`；
- 不使用 `X @ delta^T` 的稠密校正 GEMM；
- 不修改 `qweight`、`qzeros` 或 `scales`；
- 不使用权重空间残差直接替换量化权重；
- 运行时仅增加两个低秩线性计算和一次残差相加。

---

## 3. 测试环境

| 项目 | 配置 |
|---|---|
| 模型 | Qwen/Qwen2.5-0.5B |
| 量化格式 | AWQ W4A16 |
| Group size | 128 |
| AWQ kernel | GEMM |
| GPU | NVIDIA A10，23GB |
| 软件框架 | PyTorch + AutoAWQ |
| PPL 样本数 | 100 |
| PPL token 数 | 18,863 |
| PPL sequence/stride | 源测试材料未给出最终具体值 |
| 并发配置 | 最终摘要仅记录单组 tokens/s、latency 和 peak memory |

---

## 4. 测试数据

### 4.1 数据划分

原始评估数据：

```text
data/eval.jsonl：100 texts，平均约 151 words
```

为匹配校准和评估分布，将数据划分为：

| 数据文件 | 数量 | 平均长度 | 用途 |
|---|---:|---:|---|
| `data/eval_calib.jsonl` | 80 | 约 145 words | Adapter 拟合 |
| `data/eval_val.jsonl` | 20 | 约 173 words | Alpha 搜索 |
| `data/eval.jsonl` | 100 | 约 151 words | 最终 PPL 评估 |

### 4.2 数据分布问题修复

早期实验使用：

```text
data/calib.jsonl：128 条，平均约 88 words
```

而评估数据平均约 151 words。两者分布不匹配时，adapter 在校准数据上的输出误差虽然下降，但 PPL 明显恶化。

切换到与评估集同分布的数据后，单层结果由恶化转为小幅改善，为多层补偿达到最终目标奠定了基础。

### 4.3 数据独立性说明

最终评估使用完整 `eval.jsonl`，其中包含用于校准和 alpha 验证的数据。因此当前结果适合作为项目内部功能与性能验收结果，但不应视为完全独立盲测结果。

面向论文、外部发布或生产结论时，应增加同分布但完全独立的测试集。

---

## 5. 验收指标与计算方法

### 5.1 PPL 改善率

```text
PPL improvement =
(PPL_awq - PPL_qcomp) / PPL_awq × 100%
```

验收条件：

```text
PPL improvement ≥ 2%
```

等价于在当前 AWQ baseline 下：

```text
PPL_qcomp ≤ 21.5365 × 0.98 = 21.1058
```

### 5.2 吞吐保持率

```text
Throughput retention =
Throughput_qcomp / Throughput_awq × 100%
```

验收条件：

```text
Throughput retention ≥ 85%
```

---

## 6. 测试策略与测试项

| 测试编号 | 测试项 | 目的 |
|---|---|---|
| TC-01 | AWQ baseline 测试 | 建立 PPL 与性能基线 |
| TC-02 | 单层 QComp 测试 | 验证基础补偿有效性 |
| TC-03 | Rank 与 ridge λ 测试 | 评估模型容量和正则化影响 |
| TC-04 | 多层补偿测试 | 验证补偿效果是否可累加 |
| TC-05 | PPL-based alpha 搜索 | 直接以最终指标选择补偿强度 |
| TC-06 | 最终配置 PPL 验收 | 验证 PPL 改善是否达到 2% |
| TC-07 | 最终配置性能验收 | 验证吞吐保持率是否达到 85% |
| TC-08 | 实现约束检查 | 确认未生成稠密校正矩阵且未修改量化权重 |

---

## 7. 测试结果

### 7.1 TC-01：AWQ Baseline

| 指标 | 测试结果 |
|---|---:|
| PPL | 21.5365 |
| NLL | 57,904.68 |
| Tokens | 18,863 |
| Samples | 100 |
| Throughput | 20.38 tokens/s |
| Latency | 1.57 s |
| Peak memory | 0.45 GB |

结论：baseline 数据完整，可作为 QComp 验收对照。

---

### 7.2 TC-02/TC-03：单层、Rank 与 Lambda 测试

校准集：`data/eval_calib.jsonl`  
验证集：`data/eval_val.jsonl`

| 层 | Rank | Lambda | 最优 alpha | PPL | 相对 AWQ |
|---|---:|---:|---:|---:|---:|
| layer0.down_proj | 2 | 0.1 | 2.0 | 21.5183 | -0.08% |
| layer0.down_proj | 4 | 0.1 | 1.0 | 21.5174 | -0.09% |
| layer0.down_proj | 8 | 0.1 | 1.5 | **21.5011** | **-0.16%** |
| layer0.down_proj | 16 | 0.1 | 1.5 | 21.5063 | -0.14% |
| layer0.down_proj | 8 | 0.01 | 2.0 | 21.5060 | -0.14% |
| layer0.down_proj | 8 | 1.0 | 2.0 | 21.5102 | -0.12% |
| layer1.down_proj | 4 | 0.1 | 0.1 | 21.5354 | -0.00% |
| layer1.down_proj | 8 | 0.1 | 0.5 | 21.5389 | +0.01% |

测试结论：

1. 单层补偿最多仅改善约 0.16%，无法达到 2% 目标；
2. layer0 明显优于 layer1；
3. rank 从 8 增至 16 未继续改善，说明单纯增大 rank 收益有限；
4. λ 在 0.01、0.1、1.0 范围内影响较小，λ=0.1 略优且更稳定；
5. 后续应通过多层补偿累加收益，而不是继续盲目增大单层 rank。

TC-02/TC-03 结论：**功能有效，但单层性能目标未通过。**

---

### 7.3 TC-04：多层补偿测试

配置统一采用：

```text
rank = 8
ridge λ = 0.1
```

| 配置 | 补偿层数 | 最优 alpha | PPL | 相对 AWQ |
|---|---:|---:|---:|---:|
| top5_r8_lam0p1 | 5 | 0.15 | 21.4377 | -0.46% |
| top10_r8_lam0p1 | 10 | 0.15 | 21.3853 | -0.70% |
| top20_r8_lam0p1 | 20 | 0.22 | 21.2456 | -1.35% |
| **top40_r8_lam0p1** | **40** | **0.35** | **21.0379** | **-2.32%** |

测试结论：

- PPL 随补偿层数增加持续下降；
- Top-40 首次超过 2% PPL 改善目标；
- 多层累加是达到最终验收目标的关键；
- Top-40 为当前测试范围内的最终选定配置。

TC-04 结论：**通过。**

---

### 7.4 TC-05：PPL-Based Alpha 搜索

Top-40 配置的 alpha 搜索结果：

| Alpha | PPL | 相对 AWQ |
|---:|---:|---:|
| 0.08 | 21.3115 | -1.05% |
| 0.10 | 21.2679 | -1.25% |
| 0.15 | 21.1635 | -1.73% |
| 0.20 | 21.1254 | -1.91% |
| 0.25 | 21.0806 | -2.12% |
| 0.30 | 21.0528 | -2.24% |
| **0.35** | **21.0384** | **-2.31%** |
| 0.38 | 21.0540 | -2.24% |
| 0.40 | 21.0710 | -2.16% |

测试结论：

1. PPL 在 alpha=0.35 附近出现明确谷底；
2. alpha 太小时补偿不足；
3. alpha 大于最优点后，PPL 开始回升；
4. 直接依据 PPL 选择 alpha，明显优于依据输出 Frobenius 误差选择 alpha；
5. 最终复测记录的 PPL 为 21.0379，与搜索阶段的 21.0384 差异为 0.0005，不影响验收结论。

TC-05 结论：**通过。**

---

### 7.5 TC-06/TC-07：最终 PPL 与性能测试

最终配置：

```text
Selected layers = Top 40
Rank            = 8
Ridge lambda    = 0.1
Global alpha    = 0.35
```

| 指标 | AWQ baseline | AWQ + QComp | 变化 |
|---|---:|---:|---:|
| PPL | 21.5365 | **21.0379** | **-2.32%** |
| Throughput | 20.38 tokens/s | **18.89 tokens/s** | -7.3% |
| Latency | 1.57 s | 1.69 s | +0.12 s |
| Peak memory | 0.45 GB | 0.48 GB | +0.03 GB |
| Throughput retention | — | **92.7%** | 目标 ≥ 85% |

精确计算：

```text
PPL improvement
= (21.5365 - 21.0379) / 21.5365
≈ 2.315%
```

```text
Throughput retention
= 18.89 / 20.38
≈ 92.689%
```

由表中数据计算：

```text
Latency increase ≈ 7.64%
Peak-memory increase ≈ 6.67%
```

验收判断：

| 验收项 | 目标 | 实际 | 结果 |
|---|---:|---:|---|
| PPL improvement | ≥ 2% | 2.315% | **PASS** |
| Throughput retention | ≥ 85% | 92.689% | **PASS** |

TC-06/TC-07 结论：**通过。**

---

### 7.6 TC-08：实现约束检查

| 检查项 | 结果 |
|---|---|
| 是否物化 `A @ B` 稠密矩阵 | 否 |
| 是否执行稠密校正 GEMM | 否 |
| 是否修改 AWQ packed 权重 | 否 |
| 是否保留 A/B 分解形式 | 是 |
| 是否通过低秩双线性层注入 | 是 |
| 是否保持 AWQ 原始前向计算 | 是 |

TC-08 结论：**通过。**

---

## 8. 总体验收结果

| 测试类别 | 结果 |
|---|---|
| 基础算法功能 | 通过 |
| 单层补偿有效性 | 有效，但不足以单独达到最终目标 |
| 多层补偿 | 通过 |
| PPL-based alpha 搜索 | 通过 |
| PPL 指标 | 通过 |
| 吞吐保持率 | 通过 |
| 结构约束 | 通过 |
| 总体验收 | **PASS** |

最终结论：

> QComp 在 Qwen2.5-0.5B AWQ W4A16 模型上，采用 Top-40、rank=8、
> ridge λ=0.1、alpha=0.35 配置后，将 PPL 从 21.5365 降至 21.0379，
> 改善约 2.315%；吞吐由 20.38 tokens/s 降至 18.89 tokens/s，
> 保持率约 92.689%。PPL 改善和吞吐保持率两项指标均达到验收要求，
> 测试结论为通过。

---

## 9. 关键发现

### 9.1 校准数据分布匹配是必要条件

分布不匹配时，输出误差下降并不能保证 PPL 改善。将校准数据切换为与评估数据同分布后，单层结果由明显恶化转为小幅改善。

### 9.2 输出重构误差不是可靠的最终选择指标

最小化：

```text
||T - alpha * X B^T A^T||_F
```

不能保证语言模型 PPL 最优。最终采用 PPL-based alpha 搜索，直接按目标指标选择补偿强度。

### 9.3 多层补偿具有累加效果

单层最佳仅改善 0.16%，Top-20 改善 1.35%，Top-40 改善 2.32%。测试数据表明，多层累加是达到 2% 目标的主要来源。

### 9.4 性能开销在验收范围内

Top-40 为 40 个量化层额外增加 rank=8 的两个低秩线性计算。最终吞吐损失约 7.3%，吞吐保持率仍为 92.7%，高于 85% 门槛。

---

## 10. 失败实验与问题闭环

| 阶段 | 方法 | 测试结果 | 主要问题 | 修复措施 |
|---|---|---:|---|---|
| Round 1 | Weight-space `W_fp-W_awq` + FP16 激活 | PPL 35.48，+64.75% | 激活来源和拟合空间不匹配 | 改为 AWQ 输入激活与 output-space target |
| Round 2 | Output-space lstsq，8 条文本 | PPL 28.69，+33.20% | 样本少、无正则、严重过拟合 | 增加校准量并使用 ridge |
| Round 3 | Ridge + 64 条文本，但数据分布不匹配 | rank2 PPL 189.82，+781% | calib 与 eval 分布不同 | 使用同分布校准数据 |
| Final | 同分布数据 + ridge + PPL alpha + Top-40 | PPL 21.0379，-2.32% | 已达到目标 | 固化配置和复现流程 |

---

## 11. 数据一致性说明

项目已有两份结果材料中存在两组最终性能记录：

| 记录来源 | QComp PPL | QComp throughput | Retention |
|---|---:|---:|---:|
| 较早实验汇总 | 21.0384 | 17.88 tokens/s | 87.7% |
| 最新技术报告最终复测 | 21.0379 | 18.89 tokens/s | 92.7% |

本报告采用**最新技术报告中的最终复测数据**作为主验收结果，即：

```text
PPL = 21.0379
Throughput = 18.89 tokens/s
Retention = 92.7%
```

较早记录即使单独采用，也满足：

```text
PPL improvement > 2%
Throughput retention > 85%
```

因此两组记录不改变项目是否通过验收的结论。但正式发布前，应将最终 JSON、Markdown、README 和复现脚本统一到同一次测试运行，并记录 Git commit、配置哈希、adapter 哈希和运行时间。

---

## 12. 风险与限制

### 12.1 校准集与评估集存在重叠

当前使用 `eval.jsonl` 的 80% 作为校准集，同时又在完整 `eval.jsonl` 上报告最终 PPL。该设置可能使结果偏乐观。

建议在正式对外结论中增加：

```text
同分布、完全独立、未参与拟合和 alpha 搜索的 test.jsonl
```

### 12.2 测试规模有限

当前 PPL 基于：

```text
100 samples
18,863 tokens
```

尚未验证更大数据集上的稳定性。

### 12.3 缺少重复实验统计

现有材料未给出多次重复测试的均值、标准差、P50/P90 或置信区间。吞吐和时延可能受 GPU 状态、温度、时钟及后台进程影响。

### 12.4 模型规模有限

当前仅验证 Qwen2.5-0.5B。结论不能直接外推至 7B、14B 或更大模型。

### 12.5 Alpha 为全局共享

当前 40 层共用一个 alpha。逐层 alpha、分组 alpha 或受约束的联合优化可能进一步改善效果，但尚未测试。

---

## 13. 建议的发布前补充测试

以下项目不影响本轮内部验收结论，但建议在正式发布前完成：

1. 使用完全独立的同分布测试集重新评估 PPL；
2. 固定随机种子并运行不少于 3 次 PPL 与 benchmark；
3. 报告吞吐、时延的均值、标准差和 P50/P90；
4. 保存统一的 `FINAL_RESULT.json`；
5. 记录 Git commit、配置哈希、数据哈希和 adapter SHA-256；
6. 验证一键复现脚本在干净环境中可运行；
7. 在至少一个更大模型上验证趋势是否成立。

---

## 14. 复现命令

### 14.1 一键复现

```bash
bash scripts/run_final_qcomp.sh
```

### 14.2 手动复现

```bash
# 恢复 GPU，并使用本地缓存
bash restore_gpu.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# AWQ baseline PPL
python3 -m a10_quant_comp.cli eval-awq-ppl \
  --config configs/mvp_qwen.yaml \
  --overwrite

# AWQ + QComp PPL
python3 -m a10_quant_comp.cli eval-qcomp-ppl \
  --config configs/mvp_qwen.yaml \
  --adapter artifacts/final/qcomp_top40_r8_lam0p1 \
  --alpha-override 0.35 \
  --eval-path data/eval.jsonl \
  --overwrite

# AWQ benchmark
python3 -m a10_quant_comp.cli benchmark-awq \
  --config configs/mvp_qwen.yaml \
  --overwrite

# QComp benchmark
python3 -m a10_quant_comp.cli benchmark-qcomp \
  --config configs/mvp_qwen.yaml \
  --adapter artifacts/final/qcomp_top40_r8_lam0p1 \
  --overwrite
```

---

## 15. 测试产物索引

| 文件或目录 | 说明 |
|---|---|
| `configs/final_qcomp.yaml` | 冻结后的最终配置 |
| `artifacts/final/qcomp_top40_r8_lam0p1/` | 最终 Top-40 QComp adapter |
| `artifacts/final/RESULT.md` | 最终结果摘要 |
| `scripts/run_final_qcomp.sh` | 一键复现脚本 |
| `results/qcomp_search.csv` | 参数与实验结果汇总 |
| `artifacts/c5fc6a81f2f7/qcomp/qcomp_layer_scores.csv` | 候选层评分 |
| `a10_quant_comp/qcomp/mvp.py` | QComp 核心拟合逻辑 |
| `a10_quant_comp/awq_integration.py` | AWQ 运行时补偿注入 |
| `a10_quant_comp/cli.py` | 测试与构建命令入口 |
| `QComp_Technical_Report.md` | 方法与实验技术报告 |

---

## 16. 最终签署结论

**测试状态：PASS**

QComp 在当前测试环境、数据和配置下，满足项目定义的两项核心验收要求：

```text
PPL improvement >= 2%        ：满足
Throughput retention >= 85%  ：满足
```

建议将以下配置冻结为当前项目发布候选版本：

```yaml
model: Qwen/Qwen2.5-0.5B

quantization:
  method: AWQ
  bits: 4
  activation_bits: 16
  group_size: 128

qcomp:
  selected_layers: top40
  rank: 8
  ridge_lambda: 0.1
  alpha: 0.35

result:
  awq_ppl: 21.5365
  qcomp_ppl: 21.0379
  ppl_improvement_percent: 2.315
  awq_throughput_tokens_per_second: 20.38
  qcomp_throughput_tokens_per_second: 18.89
  throughput_retention_percent: 92.689
  status: PASS
```
