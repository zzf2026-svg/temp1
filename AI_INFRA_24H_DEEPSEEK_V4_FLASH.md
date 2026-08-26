# 24 小时 DeepSeek V4 Flash AI Infra 实战手册

> 实验环境：2 台服务器，每台 2 张标称 48 GiB 的 RTX 4090，共 4 卡、约 192 GiB 显存。  
> 实验周期：24 小时。  
> 主线目标：直接使用 DeepSeek V4 Flash 官方低精度权重，不自行量化。  
> 长期目标：建立生产级 AI Infra 工程师所需的分析、部署、测量、排障和工程化能力。

---

## 1. 最终目标

这一天的目标不是机械地执行一条 `vllm serve` 命令，而是完成一次可复现、可测量、可解释的分布式大模型推理实验。

24 小时结束时，应当得到以下成果：

1. 一份准确的硬件、拓扑、磁盘和网络基线报告。
2. 一套两节点四 GPU 的模型下载、集群启动和服务启动脚本。
3. 一个 OpenAI API 兼容的推理端点；如果 V4 Flash 因硬件或内核限制无法启动，则要有证据充分的根因报告和已跑通的替代模型。
4. 一份包含 TTFT、TPOT、吞吐、P50/P95/P99 延迟、错误率和资源利用率的压测报告。
5. 一份容量边界说明，包括安全上下文长度、安全并发和显存余量。
6. 至少三次故障演练及恢复记录。
7. 一份可交给其他工程师复现的 README、配置、日志和复盘文档。

### 成功的定义

实验分为三档成功：

- **A级：** V4 Flash 双机四卡成功提供稳定服务，并完成压测、监控与故障演练。
- **B级：** V4 Flash 成功加载并生成结果，但受吞吐或稳定性限制；同时完整记录瓶颈并用较小模型走通生产闭环。
- **C级：** V4 Flash 因硬件、格式、内核或网络限制无法运行，但在规定止损时间内定位根因，并用替代模型完成部署、压测、监控和故障演练。

C 级不是失败。生产级工程师的重要能力之一，就是尽早证明方案不可行、保留证据并切换到可交付路径。

---

## 2. 现实约束与关键风险

DeepSeek V4 Flash 是 MoE 模型。官方模型规模约为 284B 总参数、13B 激活参数，官方低精度 checkpoint 约 149 GiB。总显存约 192 GiB 并不代表一定能运行，因为还需要：

- CUDA context 和推理框架运行时显存；
- 通信缓冲区；
- KV Cache；
- CUDA Graph 或编译缓存；
- 模型加载期间的临时空间；
- 显存碎片余量。

还需要特别注意：标准 RTX 4090 通常为 24 GiB。租赁商标注的“4090 48G”可能是改装卡、特殊版本或识别名称不准确，必须用 `nvidia-smi` 验证实际型号和显存。

主要风险按优先级排列：

1. 官方低精度格式所需的 FP4/FP8/MoE kernel 不支持 Ada 架构。
2. 149 GiB 权重加运行时开销超过 192 GiB 总显存。
3. 两张 4090 之间没有 NVLink，节点内通信只走 PCIe。
4. 两节点之间网络过慢，跨节点并行造成严重延迟。
5. 模型下载耗时过长，吞噬租期。
6. 主机内存、磁盘空间或磁盘吞吐不足。
7. vLLM、PyTorch、CUDA、驱动和模型代码版本不兼容。

因此，本实验必须采用“先验证、后扩展；先最小可用、后性能优化”的推进方式。

---

## 3. 你将学到什么

### 3.1 GPU 与服务器基础

你将能够：

- 识别 GPU 型号、显存、驱动、功耗、频率和温度状态；
- 理解 PCIe、NVLink、NUMA 对多卡推理的影响；
- 使用 `nvidia-smi topo -m` 分析 GPU、CPU 和网卡拓扑；
- 判断瓶颈来自计算、显存容量、显存带宽还是通信；
- 估算权重、KV Cache、运行时开销和安全显存余量。

### 3.2 分布式推理

你将理解并能够选择：

- Tensor Parallel（TP）：切分单层计算，通信频繁；
- Pipeline Parallel（PP）：按层切分，跨节点通信相对少，但可能产生流水线气泡；
- Data Parallel（DP）：复制模型，提高吞吐，需要模型能在每个副本中装下；
- Expert Parallel（EP）：针对 MoE 专家进行切分和路由；
- NCCL collective、Ray worker placement 和跨节点进程组织方式。

本环境的首选假设是：节点内 `TP=2`，节点间 `PP=2`。这是需要实验验证的假设，不是预设结论。

### 3.3 LLM 推理原理

你将能够解释：

- Prefill 与 Decode 的计算特征；
- TTFT 和 TPOT 为什么对应不同瓶颈；
- Continuous Batching 如何提高吞吐；
- PagedAttention 和 KV Cache 如何影响显存；
- 上下文长度、并发和吞吐之间的关系；
- Chunked Prefill、Prefix Caching、CUDA Graph 的用途与代价；
- MoE 的总参数量、激活参数量和专家通信之间的区别。

### 3.4 生产服务能力

你将完成：

- OpenAI 兼容接口；
- 健康检查和启动就绪检查；
- 请求超时、并发限制和背压；
- 指标、日志和 GPU 监控；
- 容量规划和压测；
- 故障检测、恢复和复盘；
- 配置版本化与一键复现。

### 3.5 工程判断能力

最重要的训练不是记参数，而是形成以下工作方式：

1. 先定义成功指标。
2. 先测基线，再做修改。
3. 一次只改变一个关键变量。
4. 所有结论都附带日志或指标证据。
5. 设置时间止损线，避免无期限排错。
6. 区分实验可运行与生产可用。

### 3.6 完成流程后的能力边界

走完本流程后，你不会因为一次 24 小时实验就自动成为生产级 AI Infra 工程师，但应当完成从“会调用模型”到“能对一个推理系统作出工程判断”的跃迁。

你应该能够独立完成：

- 根据模型权重格式、GPU 显存和 KV Cache 预算判断模型是否可能装下；
- 根据 GPU 拓扑和节点网络为模型选择 TP、PP、DP、EP 的初始组合；
- 区分模型加载失败、CUDA kernel 不兼容、OOM、NCCL hang 和服务过载；
- 设计压测矩阵，而不是只观察一次 tokens/s；
- 用 TTFT、TPOT、吞吐、P99 和错误率评价服务；
- 找到最大安全上下文和最大稳定并发；
- 建立最小可用的指标、日志、健康检查、限流和超时；
- 设计并执行故障演练，计算 MTTD 和 MTTR；
- 写出带数据的容量建议、风险说明和扩容方案；
- 将实验变成其他工程师可以复现的脚本和文档。

你还不能仅凭这一天宣称已经掌握：

- Kubernetes 大规模 GPU 调度；
- 多租户隔离、计费和配额系统；
- 数十到数百节点的 RDMA 网络运维；
- 完整模型训练、FSDP/ZeRO 和 checkpoint 容错；
- 跨地域高可用、供应链安全和长期 on-call。

这些内容需要后续项目和生产值班经验。本实验的作用是建立正确的底层模型和工作方法。

### 3.7 完成后应该能回答的问题

#### 资源与可行性

1. 为什么 149 GiB 权重不能简单地认为能装入 192 GiB 总显存？
2. 权重、KV Cache、激活、通信缓冲和 CUDA context 分别占用什么资源？
3. 为什么 1M 上下文能力不代表当前硬件可以服务 1M token？
4. 当前瓶颈是显存容量、显存带宽、算力、PCIe、节点网络还是磁盘？证据是什么？
5. 为什么必须核实“4090 48G”的真实型号、Compute Capability 和 kernel 支持？

#### 并行与分布式

6. TP、PP、DP、EP 各自切分什么，通信发生在哪里？
7. 为什么 TP 通常优先放在高速互联域内，而 PP 更容易跨节点？
8. `TP=2, PP=2` 与 `TP=4, PP=1` 的通信和延迟差异是什么？
9. 为什么 DP 能提高吞吐，却不能帮助一个装不下的模型装入显存？
10. MoE 模型为什么可能需要 EP？专家负载不均衡会造成什么问题？
11. `world_size`、模型副本数和每副本 GPU 数之间是什么关系？
12. NCCL AllReduce、AllGather、ReduceScatter 和 All-to-All 分别常见于哪些并行模式？

#### 推理性能

13. Prefill 和 Decode 为什么表现出不同的算力、带宽和通信特征？
14. TTFT、TPOT、吞吐和 P99 为什么不能只选一个作为性能结论？
15. 提高并发为什么可能提高总吞吐，同时恶化用户延迟？
16. `max-num-seqs`、`max-num-batched-tokens` 和上下文长度如何影响显存及排队？
17. 如何通过对照实验判断服务是计算受限、内存带宽受限还是通信受限？

#### 生产可靠性

18. `/health` 和 `/ready` 为什么不能是同一个含义？
19. 一个跨两节点的模型副本能否算高可用？为什么？
20. 节点失联、NCCL hang、OOM 和过载在指标与用户侧分别表现为什么？
21. 什么情况下应该排队、拒绝、降级或扩容？
22. 如何根据峰值流量、目标 P99 和单副本吞吐计算所需副本数及余量？
23. 如何证明一次参数优化是真实收益，而不是预热、缓存或样本差异？
24. 什么证据足以支持 Go/No-Go 决策和生产上线结论？

如果这些问题只能背定义、不能结合本次日志和指标回答，说明流程还没有真正走完。

---

## 3A. 推理并行策略详解

并行不是“GPU 数量参数”，而是决定模型状态如何放置、每一步产生什么通信以及故障边界在哪里。生产部署需要同时考虑：模型能否装下、单请求延迟、总吞吐、网络拓扑和框架支持。

### 3A.1 总体关系

对常见稠密模型部署，可以先使用以下近似关系理解资源：

```text
每个模型副本使用的 GPU 数 ≈ TP × PP
总 GPU 数 ≈ DP × TP × PP
```

对 MoE 模型，EP 与 TP/DP 的关系由框架实现决定，不能机械地再乘一次。某些实现让 attention 使用 DP/TP，而 MoE 层使用 EP；必须查看当前 vLLM 版本实际建立的 process group。

### 3A.2 Tensor Parallel（TP，张量并行）

**切分对象：** 同一层中的矩阵权重和矩阵计算。

**典型行为：** 每生成一个 token，多层之间会频繁发生 collective 通信，常见为 AllReduce、AllGather 或 ReduceScatter。

**优势：**

- 降低每张 GPU 承担的权重和部分计算；
- 所有 GPU 同时处理同一批请求；
- 单机高速互联下通常是大模型推理的首选。

**代价：**

- 对 GPU 间带宽和延迟非常敏感；
- TP 扩大后，通信可能吞噬计算收益；
- 无 NVLink 的 4090 通过 PCIe 通信，收益可能明显低于数据中心 GPU。

**本实验要回答：** 单节点 TP=2 的 NCCL 带宽是否足够？跨节点 TP=4 是否因网络通信而不可接受？

### 3A.3 Pipeline Parallel（PP，流水线并行）

**切分对象：** 将连续的模型层分配给不同 pipeline stage。

**典型行为：** 阶段边界传输激活；不像 TP 那样在每层内进行同等频率的 collective，但会产生流水线气泡和阶段等待。

**优势：**

- 可以让单节点装不下的模型跨节点部署；
- 跨节点通信量通常比跨节点 TP 更容易控制；
- 适合节点之间网络弱于节点内部互联的拓扑。

**代价：**

- 单请求或小 batch 时 pipeline bubble 明显；
- stage 切分不均衡会由最慢 stage 决定吞吐；
- 模型结构或框架可能限制可切分位置；
- MoE 层分布不均可能让阶段负载失衡。

**本实验要回答：** `TP=2, PP=2` 能否把通信主要限制在节点内，同时让两台机器的 stage 时间接近？

### 3A.4 Data Parallel（DP，数据并行）

**切分对象：** 不切模型；每个 DP rank 持有一个完整模型副本，分别处理不同请求。

**优势：**

- 最直接地提高总请求吞吐；
- 副本之间可隔离故障；
- 配合负载均衡可实现滚动升级和高可用。

**代价：**

- 每个副本都必须能完整容纳模型；
- 权重显存和模型加载成本随副本数重复；
- 热点 prompt、长短请求混合可能导致副本负载不均。

**对本实验的结论：** V4 Flash 很可能需要全部四张 GPU 才能容纳一个副本，因此当前没有 DP 空间，也就没有真正的模型级高可用。替代 32B 模型则可以尝试两台机器各一个 TP=2 副本，学习负载均衡与容错。

### 3A.5 Expert Parallel（EP，专家并行）

**切分对象：** 将 MoE 专家分散到不同 GPU/rank，token 根据 router 结果发送到对应专家。

**典型行为：** token dispatch 和 combine 通常涉及 All-to-All 通信。

**优势：**

- 利用 MoE 每个 token 只激活部分专家的特点；
- 分散大量专家权重；
- 合适硬件和通信 backend 下可以提高 MoE 部署效率。

**代价：**

- All-to-All 对网络非常敏感；
- router 产生的专家热点会形成 straggler；
- 需要专用 MoE kernel、通信 backend 和可能的 Expert Parallel Load Balancing；
- 4090 与普通以太网未必支持官方高性能路径。

**本实验要回答：** 当前 vLLM/模型/GPU 是否支持 EP 所需 backend？开启 EP 后 GPU 间 token 分布、网络流量和尾延迟是否改善？

### 3A.6 Sequence、Context 与数据预处理并行

这些并行在长上下文或训练系统中常见，但不应为了覆盖术语而强行加入本次启动命令：

- **Sequence Parallel：** 沿序列维度切分部分激活或计算，常与 TP 配套降低激活内存；具体支持取决于框架和算子。
- **Context Parallel：** 将长上下文 token 分散到多个 rank，解决超长序列的计算和 KV/激活压力；需要注意 attention 通信。
- **Prefill/Decode Disaggregation：** 严格说不是传统模型并行，而是将 prefill 与 decode 放到不同 worker 池，并传输 KV Cache，适合两阶段资源特征差异明显的大规模服务。

本次只有四张消费级 GPU，优先把 TP/PP 和基本服务闭环做透。Sequence/Context Parallel 与 PD 分离应作为后续实验，而不是第一天的必要项。

### 3A.7 训练并行与推理并行不要混淆

生产级 AI Infra 还会遇到以下训练技术：

- **DDP：** 每张 GPU 保存完整模型和 optimizer 状态，反向传播时同步梯度；
- **FSDP/ZeRO：** 分片参数、梯度和 optimizer 状态；
- **Tensor/Pipeline/Expert Parallel：** 也可以用于训练，但同时存在 forward、backward 和 optimizer 的额外通信与内存；
- **数据流水线并行：** 数据读取、tokenize、shuffle 和预取。

本手册是推理部署手册。这里的 DP 指请求级模型副本，不等同于训练 DDP 的梯度同步行为。

### 3A.8 四张 GPU 上可比较的拓扑

| 方案 | 布局 | 能否帮助模型装下 | 通信特征 | 适合验证什么 |
|---|---|---:|---|---|
| A | TP=2, PP=2 | 是 | 节点内 TP，节点间 stage 激活 | 推荐的 V4 初始方案 |
| B | TP=4, PP=1 | 是 | TP collective 跨节点 | 量化跨节点 TP 代价 |
| C | TP=1, PP=4 | 是 | 四个 pipeline stage | PP 气泡与 stage 平衡 |
| D | DP=2, 每副本 TP=2 | 每副本需能装下 | 副本间几乎无模型通信 | 小模型吞吐与高可用 |
| E | TP/DP + EP | 取决于实现 | MoE All-to-All | 仅在 kernel/backend 支持时测试 |

不要在 V4 无法启动时依次盲试全部组合。先用较小模型验证 A/B/C 的通信行为，再决定 V4 是否值得进行第二种拓扑实验。

### 3A.9 并行策略决策顺序

```text
模型能否在一张 GPU 装下？
├── 能
│   ├── 目标是降低单请求延迟：先单卡基线，再判断 TP 是否有收益
│   └── 目标是提高总吞吐/高可用：优先 DP
└── 不能
    ├── 能否在单节点装下？
    │   ├── 能：优先测试节点内 TP；无高速互联时也测试 PP
    │   └── 不能：采用跨节点 PP + 节点内 TP
    └── 如果是 MoE：在 backend 与网络支持时评估 EP
```

每个选择都必须同时检查：

1. 显存是否满足；
2. collective 是否跨越慢速链路；
3. 框架和 kernel 是否支持；
4. TTFT、TPOT、吞吐和 P99 是否达到目标；
5. 任一 rank 失败时的故障域有多大。

### 3A.10 本次必须完成的并行实验

如果 V4 Flash 能稳定启动，至少完成：

1. `TP=2, PP=2` 的功能与性能基线；
2. 采集每张 GPU 的显存、利用率和节点间网络吞吐；
3. 分别使用短 prefill/长 decode 与长 prefill/短 decode 请求观察差异；
4. 若剩余时间和框架支持允许，再与 `TP=4, PP=1` 做一次小规模对照。

如果 V4 Flash 不能启动，则用较小模型完成：

1. 单机 TP=1 与 TP=2 对比；
2. 双机 `TP=2, PP=2` 或缩小后的等价 PP 实验；
3. 两个 DP 副本与单个副本的吞吐、故障隔离对比；
4. 用 NCCL 和网络指标解释结果，而不是只报告“哪个更快”。

---

## 4. 24 小时总时间表

| 时间 | 阶段 | 核心任务 | 必须产出 |
|---|---|---|---|
| 00:00–00:30 | 启动 | 两节点同时开始下载，采集系统信息 | 原始环境信息 |
| 00:30–01:30 | 硬件基线 | GPU、CPU、内存、磁盘、拓扑测试 | `hardware.md` |
| 01:30–02:30 | 网络基线 | iperf3、单机及双机 NCCL 测试 | `network.md`、NCCL 日志 |
| 02:30–03:00 | 可行性闸门 | 判断是否继续 V4 Flash | Go/No-Go 记录 |
| 03:00–05:00 | 软件栈冒烟 | 用较小模型验证 vLLM、GPU、API | 可调用的测试端点 |
| 05:00–09:00 | V4 主部署 | 双机四卡加载官方低精度模型 | 首个响应或完整错误日志 |
| 09:00–11:00 | 第二轮排障 | 基于证据修改一次配置 | 成功结果或根因报告 |
| 11:00–14:00 | 性能压测 | 并发、长短输入输出矩阵 | 原始结果和汇总表 |
| 14:00–16:00 | 参数实验 | 至少三组单变量对照实验 | 调优结论 |
| 16:00–18:00 | 可观测性 | GPU、请求、系统与日志监控 | 仪表盘或采集文件 |
| 18:00–20:00 | 故障演练 | Worker、网络、过载三类故障 | 故障复盘 |
| 20:00–22:00 | 服务化 | 健康检查、限流、超时、入口 | 生产化端点 |
| 22:00–24:00 | 归档 | 脚本、指标、架构图和总结 | 可复现实验仓库 |

建议预留两次 30–45 分钟休息。最后两小时必须用于归档，不要继续进行没有止损线的尝试。

---

## 5. 推荐目录结构

在两台机器上使用一致的目录：

```text
ai-infra-lab/
├── README.md
├── docs/
│   ├── hardware.md
│   ├── network.md
│   ├── architecture.md
│   ├── capacity.md
│   └── final-report.md
├── scripts/
│   ├── inspect.sh
│   ├── download-model.sh
│   ├── start-head.sh
│   ├── start-worker.sh
│   ├── serve-v4.sh
│   ├── serve-fallback.sh
│   ├── smoke-test.sh
│   └── benchmark.sh
├── configs/
├── logs/
│   ├── node1/
│   └── node2/
├── results/
│   ├── nccl/
│   ├── benchmarks/
│   └── metrics/
└── postmortems/
```

所有关键命令都应进入脚本，禁止只保留在 shell history 中。

---

## 6. 第 0–3 小时：基线与可行性判断

### 6.1 立即并行开始下载

如果两台机器没有共享高速文件系统，两台都需要下载或获得相同路径下的权重。先确认每台机器至少有 200–250 GiB 可用空间。

```bash
mkdir -p /data/models
python -m pip install -U huggingface_hub

huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /data/models/DeepSeek-V4-Flash
```

如果租赁环境没有 `/data`，替换为实际高速数据盘路径。记录：

- 下载开始与结束时间；
- 平均下载速度；
- 权重总大小；
- 两节点模型目录的文件数量和校验结果。

### 6.2 系统信息采集

两台机器分别执行并保存输出：

```bash
date -u
hostname
uname -a
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id \
  --format=csv
nvidia-smi topo -m
lscpu
numactl --hardware
free -h
df -h
lsblk
ip -br addr
```

GPU 持续状态采集：

```bash
nvidia-smi dmon -s pucvmet -d 1
```

至少回答这些问题：

- 实际 GPU 名称和显存是多少？
- 每个节点两张 GPU 是否位于同一 NUMA 节点？
- GPU 之间经过什么 PCIe 路径？
- 网卡与 GPU/CPU 的相对拓扑是什么？
- 主机内存能否支撑模型加载？
- 模型盘是否为本地 NVMe？

### 6.3 节点网络基线

节点 1：

```bash
iperf3 -s
```

节点 2：

```bash
iperf3 -c <NODE1_IP> -P 8 -t 30
iperf3 -c <NODE1_IP> -P 8 -t 30 -R
```

经验判断：

- 1 GbE：不适合认真挑战跨节点大模型并行；
- 10 GbE：适合学习，性能大概率较差；
- 25 GbE：可以实验，但要关注尾延迟；
- 100 GbE 及以上：更接近生产级多机推理环境；
- 有 InfiniBand/RDMA：进一步验证 GPUDirect RDMA。

### 6.4 NCCL 基线

使用 `nccl-tests` 分别测试：

1. 单节点双卡；
2. 双节点四卡；
3. 小消息和大消息下的带宽。

典型测试命令：

```bash
./build/all_reduce_perf -b 8M -e 1G -f 2 -g 2
```

调试时设置：

```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=<实际高速网卡名>
```

不要在没有确认 IB 设备和驱动的情况下盲目启用 IB。保存完整 NCCL 日志，并记录算法带宽、总线带宽、错误和超时。

### 6.5 第一个 Go/No-Go 闸门

满足以下条件才进入 V4 Flash 主部署：

- 实际总显存约 192 GiB；
- 每台机器有足够的主机内存和高速磁盘空间；
- checkpoint 能在可接受时间内完成下载；
- 双节点能够互通，所需端口未被防火墙阻断；
- 单机和双机 NCCL 能完成；
- 当前软件栈能识别 CUDA GPU；
- 已接受“4090 低精度 kernel 可能不兼容”的实验风险。

在 `docs/final-report.md` 中写下判断，格式如下：

```text
时间：
结论：GO / CONDITIONAL GO / NO-GO
关键证据：
最大风险：
下一次检查时间：第 9 小时
止损条件：第 11 小时仍不能生成首个 token
```

---

## 7. 第 3–5 小时：软件栈冒烟测试

不要直接用 284B 模型验证所有变量。先用较小模型证明以下链路正常：

```text
CUDA → PyTorch → vLLM → 多 GPU → HTTP API → 客户端
```

可选模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --served-model-name deepseek-lab
```

冒烟检查：

```bash
curl --fail http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-lab",
    "messages": [
      {"role": "user", "content": "用三句话解释 TP 和 PP 的区别"}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```

验收标准：

- `/v1/models` 返回 200；
- Chat Completion 能稳定返回；
- 两张 GPU 均参与工作；
- 没有持续 OOM、Xid 或 NCCL 错误；
- 重启后能通过脚本复现。

如果冒烟测试失败，不要进入 V4 主部署。先把错误定位到驱动、CUDA、PyTorch、vLLM、模型或通信中的一层。

---

## 8. 第 5–11 小时：V4 Flash 双机四卡部署

### 8.1 初始并行策略

优先尝试：

```text
节点 1：GPU 0、1 ── TP=2 ┐
                         ├── PP=2 ── 一个模型副本
节点 2：GPU 0、1 ── TP=2 ┘
```

理由：

- TP 通信频繁，尽量限制在节点内部；
- PP 主要在阶段边界传输激活，更适合跨节点；
- 4090 通常没有 NVLink，TP=2 仍会受 PCIe 限制；
- 是否优于 TP=4 跨节点，必须通过实测确认。

### 8.2 环境一致性

两节点必须一致：

- NVIDIA 驱动兼容；
- CUDA/PyTorch/vLLM 版本相同；
- 容器镜像或 Python 环境相同；
- 模型路径相同；
- 模型文件完整；
- 环境变量和网络接口配置明确。

生产实践中优先使用固定 digest 的容器镜像。实验阶段至少记录：

```bash
python --version
python -m pip freeze
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -c 'import vllm; print(vllm.__version__)'
```

### 8.3 Ray 集群

使用一台机器作为 head，另一台作为 worker。具体命令应以当前 vLLM 版本提供的多节点启动脚本为准。启动后检查：

```bash
ray status
```

必须看到两个节点和四张 GPU。如果 worker 没有加入，不要启动模型。

### 8.4 最保守启动配置

先追求生成第一个 token，不追求长上下文或高并发：

```bash
vllm serve /data/models/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 8000 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --served-model-name deepseek-v4-flash
```

注意：这是实验起点，不保证适用于具体 vLLM 版本和 4090。模型专用 parser、低精度后端和额外参数应依据实际模型卡、vLLM 版本及错误日志补充，不能无证据地堆叠参数。

首轮只执行：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "回复 OK"}],
    "temperature": 0,
    "max_tokens": 20
  }'
```

### 8.5 两轮排障原则

第一轮：完整复现，保存所有日志，不凭感觉修改。

第二轮：根据证据只调整一至两个变量，再次验证。

错误分类：

| 现象 | 调查方向 |
|---|---|
| 模型架构无法识别 | vLLM 版本、remote code、模型文件 |
| Unsupported kernel | GPU 架构、FP4/FP8/MoE backend |
| 加载时 OOM | 权重格式、加载峰值、通信缓冲、显存碎片 |
| Ray worker pending | GPU 资源、节点地址、容器网络、placement |
| NCCL timeout | 网卡、防火墙、接口选择、IB/RDMA 配置 |
| 编译时间异常 | Triton、torch.compile、CUDA Graph、缓存 |
| 成功但极慢 | PCIe、跨节点通信、CPU offload、错误并行策略 |
| 生成内容异常 | tokenizer、chat template、reasoning parser |

每次实验记录：

```text
实验编号：
假设：
唯一修改项：
完整启动命令：
结果：
日志路径：
结论：
下一步：
```

### 8.6 第二个止损闸门

第 11 小时仍未生成首个 token：

1. 停止继续尝试随机组合参数；
2. 保存模型、框架、驱动和 NCCL 日志；
3. 写明失败发生在哪一层；
4. 切换到已经冒烟通过的较小模型；
5. 使用剩余时间完成生产服务闭环。

不要为了“模型名字”牺牲压测、监控、可靠性和复盘。这些才是可迁移的 AI Infra 能力。

---

## 9. 第 11–16 小时：压测、容量规划与调优

### 9.1 先定义指标

- **TTFT：** 从发出请求到收到第一个 token；
- **TPOT：** 第一个 token 之后，每个输出 token 的平均时间；
- **端到端延迟：** 请求开始到完整响应结束；
- **吞吐：** 每秒处理的输入、输出和总 token 数；
- **并发：** 同时在处理或排队的请求数；
- **错误率：** 非 2xx、超时、OOM 和连接错误占比；
- **P50/P95/P99：** 延迟分布，生产环境重点关注尾延迟。

### 9.2 最小压测矩阵

| 场景 | 并发 | 输入 token | 输出 token | 目的 |
|---|---:|---:|---:|---|
| S1 | 1 | 128 | 128 | 单请求基线 |
| S2 | 4 | 128 | 128 | 轻并发 |
| S3 | 16 | 128 | 128 | 吞吐与排队 |
| S4 | 4 | 2048 | 128 | Prefill 压力 |
| S5 | 4 | 128 | 1024 | Decode 压力 |
| S6 | 1 | 4096 | 32 | 上下文边界冒烟 |

每个场景：

- 预热后再测；
- 至少重复三次；
- 固定 prompt 数据和采样参数；
- 同时采集 GPU、CPU、内存和网络指标；
- 保存原始逐请求结果，不只保存平均值。

### 9.3 单变量调优

至少完成三组实验，每次只改变一个变量：

- `max-num-seqs`；
- `max-num-batched-tokens`；
- `gpu-memory-utilization`；
- Chunked Prefill 开关；
- Prefix Caching 开关；
- TP/PP 组合；
- 最大上下文长度。

结论必须使用数据表达：

```text
实验：并发 4 → 16
总吞吐：+68%
P99 TTFT：+190%
GPU 利用率：61% → 92%
错误率：0% → 0.4%
结论：批处理任务可使用并发 16，交互服务建议限制在 4–8。
```

### 9.4 容量规划

最后给出两套容量建议：

- **交互型配置：** 优先 TTFT 和 P99；
- **离线型配置：** 优先总 tokens/s。

容量报告至少包含：

```text
模型与版本：
硬件拓扑：
并行策略：
安全最大上下文：
安全最大并发：
稳定吞吐：
P95/P99 TTFT：
显存余量：
网络峰值：
拒绝请求的条件：
测量适用范围：
```

---

## 10. 第 16–18 小时：可观测性

生产级服务至少要覆盖四类信号。

### 10.1 GPU 指标

- GPU utilization；
- 显存使用量；
- 功耗、温度、频率；
- PCIe 发送/接收；
- Xid 和硬件错误。

### 10.2 推理服务指标

- Running/Waiting requests；
- Prompt/Generation tokens；
- TTFT、TPOT 和端到端延迟；
- KV Cache 使用率；
- 请求成功率、超时率和取消率。

### 10.3 系统指标

- CPU、内存和 swap；
- 磁盘吞吐与容量；
- 网卡吞吐、丢包和重传；
- 进程和容器状态。

### 10.4 日志

日志至少包含：

- 服务启动参数与版本；
- 模型加载时间线；
- Ray worker 状态；
- NCCL backend 与所选网卡；
- 请求错误和 OOM；
- 重启原因。

监控实现可以采用 DCGM Exporter + Prometheus + Grafana，也可以在短租期内使用脚本定时采集到 CSV。重点不是界面，而是能否将一次延迟尖峰关联到 GPU、队列、网络或错误日志。

---

## 11. 第 18–20 小时：故障演练

必须在可控条件下完成以下三类演练。

### 11.1 Worker 进程退出

- 触发：终止一个 worker；
- 观察：接口表现、Ray 状态、错误日志；
- 测量：检测时间、失败请求数、恢复时间；
- 思考：是否自动恢复，是否需要重启整个模型副本。

### 11.2 节点间网络异常

- 触发：在租赁平台许可范围内短暂限制实验端口或测试网卡；
- 观察：NCCL timeout、请求挂起、健康检查；
- 测量：超时是否有上限，客户端是否能及时失败；
- 思考：怎样避免无限挂起和级联故障。

### 11.3 过载

- 触发：并发逐步增加到容量边界之外；
- 观察：排队长度、P99、OOM、超时和错误率；
- 测量：服务从何时开始失去稳定性；
- 思考：限流、排队上限和背压策略。

每次使用统一模板：

```text
故障名称：
日期时间：
触发方式：
预期行为：
实际用户侧现象：
首个可观测信号：
根因证据：
恢复操作：
MTTD：
MTTR：
失败请求数：
预防措施：
```

---

## 12. 第 20–22 小时：生产服务化

实验端点和生产端点之间至少差这些能力：

### 必须完成

- `/health`：进程健康；
- `/ready`：模型已经加载且能够接收请求；
- 请求超时；
- 最大请求体和最大 token 限制；
- 并发上限；
- 明确的错误码；
- 访问日志和请求 ID；
- 服务优雅退出；
- 启动失败时非零退出码。

### 应当理解

- API 鉴权；
- 租户级限流；
- 配额与计费；
- TLS；
- 灰度发布；
- 多副本负载均衡；
- 模型版本回滚；
- Prompt 和输出中的敏感数据治理。

本次只有一个跨节点模型副本时，不要把一个 worker 当成高可用副本。任意节点失败都可能导致整个模型不可用。真正的高可用至少需要另一个完整副本，以及副本外的健康感知负载均衡。

---

## 13. 第 22–24 小时：归档与复盘

机器释放前完成以下操作：

1. 保存所有启动命令和配置。
2. 导出 `pip freeze`、容器 tag/digest、驱动和 CUDA 版本。
3. 保存 NCCL、Ray、vLLM 和系统日志。
4. 导出压测原始数据和图表。
5. 写出最终推荐参数和容量边界。
6. 清除日志中的 Hugging Face token、API key、内网凭证等敏感数据。
7. 将结果同步到租赁机器之外的持久存储。

最终报告必须回答：

- V4 Flash 是否成功？成功到哪个等级？
- 如果失败，失败发生在模型、格式、kernel、显存、调度、网络还是服务层？
- 哪些证据支持这个结论？
- 当前硬件的最大稳定能力是什么？
- TTFT、TPOT、吞吐和 P99 如何？
- 哪个参数改变带来了最大收益？
- 发生过哪些故障？检测和恢复耗时多久？
- 如果再租一天，应当换什么硬件或修改什么方案？

---

## 14. 生产级验收清单

### 基础设施

- [ ] 两节点硬件与软件版本有完整清单
- [ ] GPU/CPU/NUMA/网卡拓扑已记录
- [ ] 磁盘、TCP 和 NCCL 基线已记录
- [ ] 模型文件路径和完整性已验证
- [ ] 环境可以通过脚本或容器复现

### 推理服务

- [ ] 服务提供 OpenAI 兼容接口
- [ ] 冷启动时间已测量
- [ ] 上下文和并发安全边界已测量
- [ ] TTFT、TPOT、吞吐、P99 和错误率已测量
- [ ] 参数选择有对照实验支持

### 可观测性与可靠性

- [ ] GPU、系统、网络和服务指标可以关联
- [ ] 请求有超时、并发上限和错误处理
- [ ] 就绪检查不会在模型加载前误报成功
- [ ] 三类故障演练均有记录
- [ ] MTTD、MTTR 和失败请求数已统计

### 工程质量

- [ ] 所有关键命令进入版本化脚本
- [ ] 配置中没有硬编码密钥
- [ ] 日志中没有泄漏 token
- [ ] 另一个工程师可按照 README 复现
- [ ] 对“实验可运行”和“生产可用”作出明确区分

---

## 15. 从一天实验到生产级工程师的后续路线

一天无法直接把任何人变成生产级 AI Infra 工程师，但可以建立正确起点。后续建议按以下顺序推进。

### 第 1 阶段：单机推理与性能模型

- 熟练掌握 vLLM/SGLang 之一；
- 能计算显存预算；
- 能解释 prefill/decode、KV Cache 和 batching；
- 能用 profiler 找出瓶颈。

### 第 2 阶段：分布式推理

- 系统学习 TP、PP、DP、EP；
- 熟悉 NCCL、RDMA、NUMA 和 GPU Direct；
- 能根据拓扑设计并行策略；
- 能排查 hang、timeout 和性能退化。

### 第 3 阶段：平台工程

- Docker、Kubernetes、GPU Operator；
- KubeRay 或等价调度方案；
- 模型缓存、镜像、制品和版本管理；
- 自动扩缩容、队列和多租户隔离。

### 第 4 阶段：SRE 与生产治理

- SLI/SLO 和错误预算；
- Prometheus、Grafana、日志和 tracing；
- 灰度、回滚、故障演练和容量管理；
- 安全、权限、数据治理和成本归因。

### 第 5 阶段：训练基础设施

- PyTorch Distributed、FSDP、DeepSpeed；
- Checkpoint、容错与数据流水线；
- 训练通信分析和集群调度；
- 从推理扩展到微调及大规模训练。

---

## 16. 一天结束时应该能说出的结论

完成本手册后，你应当能够用数据回答，而不是凭感觉回答：

> 这个模型为什么能或不能在这四张 GPU 上运行？

> 当前瓶颈是权重显存、KV Cache、计算、PCIe 还是跨节点网络？

> 为什么选择 TP=2、PP=2，而不是 TP=4？

> 服务在什么上下文和并发范围内满足目标 P99？

> 一个节点或 worker 失败时，用户会看到什么，系统多久能够恢复？

> 如果需要把服务提升为真正的生产系统，还缺少哪些资源和机制？

能够可靠地回答这些问题，并提供脚本、日志和指标作为证据，才是这次 24 小时实验最有价值的结果。

---

## 17. 官方参考资料

- DeepSeek V4 Flash 模型卡：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- vLLM DeepSeek V4 Flash 配方：<https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml>
- vLLM 并行与扩展：<https://docs.vllm.ai/en/v0.20.0/serving/parallelism_scaling/>
- vLLM 多节点服务示例：<https://docs.vllm.ai/en/v0.20.1/examples/online_serving/multi-node-serving/>

执行前应再次核对模型卡和所安装 vLLM 版本的文档。模型架构支持、参数名称和低精度 kernel 会随版本变化，不应仅依赖过往命令。
