# 关于“AI Infra × 存储 × CUDA/KV Cache”交叉方向的两次回答

## 问题一：这个交叉方向很有意思，具体讲讲

对，这个方向真正有意思的地方就在于：**它不是“学一点 CUDA + 学一点 SSD”这么简单，而是围绕 LLM 的数据生命周期，把 GPU 算子、内存管理、缓存算法和存储系统串成一条链。**

现在已经不是概念了。vLLM 官方已经支持把完成的 KV blocks 从 GPU offload 到 CPU/更慢层级，并使用 pinned memory + `cudaMemcpyAsync` 与模型计算异步重叠；LMCache 则进一步把 KV Cache 做成独立缓存层，可以放 CPU、SSD、远端存储等。([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/?utm_source=chatgpt.com))

---

# 1. 先看这个方向到底在解决什么

假设一个 LLM 正在服务很多用户。

Attention 每生成一个 token，都会产生 K/V：

```text
token 1 → K1 V1
token 2 → K2 V2
token 3 → K3 V3
...
token N → KN VN
```

这些东西通常都放 GPU 显存：

```text
        GPU
┌──────────────────┐
│ Model weights    │
│                  │
│ KV Cache user A  │
│ KV Cache user B  │
│ KV Cache user C  │
│ KV Cache user D  │
└──────────────────┘
```

问题来了：

> 用户一多、上下文一长，显存主要可能不是被模型权重吃掉，而是被 KV Cache 吃掉。

举个**假想模型**：

```text
32 layers
8 KV heads
head_dim = 128
BF16
context = 100k
```

一个请求的 KV 大约：

\[
2\times32\times8\times128\times100000\times2
\]

已经大约 **12 GiB**。

其中第一个 2 是：

```text
K + V
```

最后一个 2 是：

```text
BF16 = 2 bytes
```

那 4 个这样的请求就接近：

```text
~49 GiB
```

所以你立刻发现：

> GPU 算力可能还有，但显存已经没了。

这就是这个交叉领域出现的原因。

---

# 2. 最直接的解决思路：给 KV Cache 建“仓库”

于是系统变成：

```text
                  LLM
                   │
                   ↓
             Attention Kernel
                   │
                   ↓
            ┌─────────────┐
            │ GPU HBM     │  ← 热数据
            │ KV Cache    │
            └──────┬──────┘
                   │
               PCIe / DMA
                   │
            ┌──────▼──────┐
            │ CPU DRAM    │  ← 温数据
            │ Pinned Mem  │
            └──────┬──────┘
                   │
                   ↓
            ┌─────────────┐
            │ NVMe SSD    │  ← 冷数据
            └─────────────┘
```

GPU：

> 快，但是贵、容量小。

CPU RAM：

> 慢一些，但容量大。

SSD：

> 更慢，但便宜，而且容量巨大。

这其实就是：

## LLM 的分层存储系统

vLLM 当前的 KV offloading 就是这个思路：GPU block 可以 offload 到 pinned CPU memory，需要时再 promote 回 GPU；传输通过 DMA，并尝试和模型计算异步重叠。([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/?utm_source=chatgpt.com))

LMCache甚至已经支持 CPU memory、local filesystem、Mooncake、Redis/Valkey 等多种后端。([docs.lmcache.ai](https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html?utm_source=chatgpt.com))

---

# 3. 那么 CUDA 算子从哪里插进来？

这是最关键的问题。

很多人看到：

```text
GPU → CPU → SSD
```

会觉得：

> 这不就是 memcpy 吗？跟 CUDA kernel 有什么关系？

实际上关系很大。

因为真正系统可能不是：

```text
KV Cache
↓
memcpy
↓
SSD
```

而是：

```text
FP16 KV Cache
      ↓
CUDA Kernel
      ↓
量化 / 压缩 / 重排 / Pack
      ↓
INT8 / FP8 KV
      ↓
DMA
      ↓
CPU / SSD
```

取回来则反过来：

```text
SSD
 ↓
CPU
 ↓
DMA
 ↓
GPU
 ↓
CUDA Kernel
 ↓
解压 / 反量化 / Layout转换
 ↓
Attention Kernel
```

看到没有？

这里已经出现很多 CUDA kernel 了。

---

# 4. 第一个交叉点：KV Cache 压缩 Kernel

例如原始 KV：

```text
FP16

2 bytes
2 bytes
2 bytes
2 bytes
...
```

假设 KV Cache 12 GB。

如果做：

```text
FP16
 ↓
量化 kernel
 ↓
INT8
```

可能就变成大约：

```text
6 GB
```

那么：

### GPU 显存

少占一半。

### PCIe

少传一半。

### CPU 内存

少占一半。

### SSD

少写一半。

所以一个 CUDA kernel 的优化，影响的是整个存储链：

```text
             Quant Kernel
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
GPU memory ↓            PCIe traffic ↓
        ↓                   ↓
CPU memory ↓             SSD IO ↓
```

这就是：

> **算子 × 存储**

非常典型的交叉。

---

# 5. 第二个交叉点：Layout 转换

Attention kernel 想要的数据格式，和 SSD 最适合保存的数据格式，不一定一样。

例如 GPU 算子喜欢：

```text
[layer][block][head][token][dim]
```

因为这样可以让线程：

```text
thread0 → dim0
thread1 → dim1
thread2 → dim2
...
```

连续读取。

但是存储系统可能希望：

```text
[request][block][layer][KV]
```

因为更容易：

```text
一次连续读取一个 KV block
```

于是就出现：

```text
Storage Layout

[request][block][layer][...]

        ↓

CUDA layout transform kernel

        ↓

GPU Attention Layout

[layer][block][head][token][dim]
```

这和你现在学 CUDA 里的：

- contiguous
- stride
- coalesced access
- vectorized load

就直接联系上了。

---

# 6. 第三个交叉点：搬运和计算重叠

这个尤其重要。

最笨的做法：

```text
SSD → CPU
等
等
等

CPU → GPU
等
等
等

Attention
开始计算
```

GPU 大量时间在：

> 等数据。

真正好的系统应该是：

```text
时间 ─────────────────────────────→

GPU compute:
████████████████████████████

PCIe transfer:
      ████████      ████████

SSD read:
████████      ████████
```

也就是：

> **一边计算，一边偷偷把下一批数据搬进来。**

这叫：

- asynchronous transfer
- prefetch
- pipeline
- compute/communication overlap

例如：

```text
处理 KV block 10

同时
     ↓
SSD读取 block 12

同时
     ↓
CPU→GPU 搬 block 11
```

等 block 10 算完：

```text
block 11 已经在 GPU
```

马上继续算。

所以工程师需要懂：

```text
CUDA Stream
CUDA Event
cudaMemcpyAsync
Pinned Memory
DMA
Pipeline
Double Buffer
```

你现在如果只会：

```cpp
kernel<<<grid, block>>>();
```

那还是单纯 kernel。

等你开始研究：

```cpp
cudaMemcpyAsync(...)
kernel<<<..., stream>>>(...)
cudaEventRecord(...)
```

实际上就已经逐渐进入：

> GPU Runtime / AI Infra

vLLM 当前 KV offloading 的实现思路本身就明确使用 asynchronous DMA，并尝试与模型计算并行。([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/?utm_source=chatgpt.com))

---

# 7. 第四个交叉点：Prefetch

假设 SSD 上有：

```text
KV block 001
KV block 002
KV block 003
...
KV block 10000
```

现在 Attention 马上需要：

```text
block 820
```

如果这时候才：

```text
SSD
 ↓
read
 ↓
CPU
 ↓
GPU
```

晚了。

所以系统需要预测：

> “GPU 接下来会需要哪个 block？”

例如：

```text
当前正在算：

block 818

预测：
819 → 马上用
820 → 马上用

提前：

SSD → CPU
```

甚至：

```text
CPU → GPU
```

这就叫：

## Prefetch

LMCache目前就有这种机制，可以预先把存储层中的 KV Cache 加载到 pinned CPU RAM，从而隐藏后续访问延迟。([docs.lmcache.ai](https://docs.lmcache.ai/kv_cache/cpu_ram.html?utm_source=chatgpt.com))

所以这里已经出现了：

```text
算法：
哪些数据值得预取？

系统：
什么时候搬？

CUDA：
怎么异步搬？

存储：
怎么最快读？
```

四个领域同时出现。

---

# 8. 第五个交叉点：Eviction

反过来还有个问题：

GPU 满了。

例如：

```text
GPU KV Blocks

001
002
003
004
...
999
```

必须踢出去一部分。

那踢谁？

最简单：

```text
LRU
Least Recently Used
```

很久没使用：

```text
block 27
```

那么：

```text
GPU
 ↓
CPU
```

CPU 又满：

```text
CPU
 ↓
SSD
```

于是：

```text
                 HOT
                  ↓

GPU       ███████████
              ↓
CPU       █████████████████
              ↓
SSD       █████████████████████████

                  ↓
                 COLD
```

这已经非常像传统的：

```text
CPU Cache
数据库 Buffer Pool
操作系统 Page Cache
```

所以传统存储工程师的很多思想，在 LLM 时代突然又有用了。

---

# 9. 第六个交叉点更硬核：SSD 直接跟 GPU 搬数据

传统路径：

```text
SSD
 ↓
CPU memory
 ↓
GPU memory
```

这里 CPU RAM 是中转站。

但还有一类技术希望：

```text
SSD
 ↓
GPU
```

尽量减少 CPU bounce buffer。

这就是 NVIDIA **GPUDirect Storage（GDS）** 想解决的问题：建立 storage 和 GPU memory 之间的直接 DMA 数据路径，从而降低 CPU 参与程度和中间拷贝。([docs.nvidia.com](https://docs.nvidia.com/gpudirect-storage/overview-guide/?utm_source=chatgpt.com))

所以：

```text
普通模式

NVMe SSD
   ↓
CPU DRAM
   ↓
GPU HBM


GDS

NVMe SSD
   ─────→
       GPU HBM
```

这时候存储公司就非常有参与空间了：

```text
NVMe SSD
PCIe
DMA
文件系统
cuFile
GPU Memory
CUDA
```

已经真正变成：

> **存储硬件 × CUDA × AI Infra**

LMCache现在也已经提供 GDS backend，用于 KV Cache 与存储之间的数据路径。([docs.lmcache.ai](https://docs.lmcache.ai/kv_cache/storage_backends/gds.html?utm_source=chatgpt.com))

---

# 10. 所以这类工程师一天到底在干什么？

可能并不是：

```text
今天写 RMSNorm
明天写 SiLU
后天写 MatMul
```

而更可能是：

### 问题一

> KV Cache FP16 太大。

做：

```text
KV FP16
 ↓
CUDA quant kernel
 ↓
INT8 KV
```

Benchmark：

```text
压缩速度：xxx GB/s
PCIe流量下降多少
显存下降多少
Attention性能下降多少
```

---

### 问题二

> SSD → GPU 太慢。

分析：

```text
NVMe bandwidth
PCIe bandwidth
CPU memcpy
Pinned memory
DMA
```

然后做：

```text
async IO
+
cudaMemcpyAsync
+
double buffer
+
prefetch
```

目标：

```text
让 IO latency 隐藏在 compute 后面
```

---

### 问题三

> Attention 在等 KV Cache。

可能改：

```text
Attention kernel
```

让它：

```text
block 0 ready → 先算 block 0

block 1
后台加载

block 2
后台预取
```

这就是：

> storage-aware kernel scheduling

---

### 问题四

> KV 存 SSD 格式和 GPU 格式不一样。

于是写：

```text
unpack kernel
transpose kernel
layout transform kernel
dequant kernel
```

甚至融合：

```text
Load
 +
Dequant
 +
Layout transform
 +
Attention

↓ fuse ↓

一个 kernel
```

这就非常 CUDA 了。

---

# 11. 这个方向的技术栈其实长这样

你可以把它分成五层：

```text
═════════════════════════════════

       LLM / Transformer
   Attention / MoE / Prefill
               │
═══════════════▼═════════════════

         AI Runtime
 vLLM / SGLang / LMCache
 KV Block Manager / Scheduler
               │
═══════════════▼═════════════════

        CUDA Kernel
 Attention
 Quant / Dequant
 Pack / Unpack
 Layout Transform
               │
═══════════════▼═════════════════

      Memory / Transport
 CUDA DMA
 Pinned Memory
 PCIe / NVLink / RDMA
               │
═══════════════▼═════════════════

           Storage
    DRAM / CXL / NVMe SSD
   File system / Object Storage

═════════════════════════════════
```

一般一个人不会五层全部精通。

真正有价值的是：

> **主攻一层，同时能看懂相邻两层。**

---

# 12. 如果你是 CUDA 算子出身，最合适站在哪里？

我认为你的切入点应该是：

```text
        LLM Runtime
            ↑
            │
    ┌───────────────┐
    │ CUDA Kernel   │  ← 主战场
    └───────────────┘
            │
            ↓
 Memory / Storage
```

也就是：

> **Memory-aware CUDA Kernel / LLM Runtime**

而不是一开始跑去啃：

```text
NAND Flash Cell
ECC
FTL
SSD firmware
```

后者完全是另一条技术树。

---

# 13. 你特别适合做一个这样的项目

按照你现在单卡 Qwen + CUDA 学习的路线，我反而建议以后做：

## 「KV Cache Tiered Storage Lab」

第一版：

```text
Qwen
 ↓
KV Cache
 ↓
GPU
```

先测：

```text
显存占用
TTFT
tokens/s
```

第二版：

```text
GPU KV
 ↕
Pinned CPU Memory
```

实现：

```text
cudaMemcpyAsync
CUDA Stream
double buffer
```

第三版：

```text
FP16 KV
 ↓
自己写 CUDA kernel
 ↓
INT8 KV
 ↓
CPU
```

也就是：

> 自己写一个 KV quant/dequant kernel。

第四版：

```text
GPU
 ↕
CPU
 ↕
NVMe SSD
```

加入：

```text
block manager
LRU
prefetch
eviction
```

第五版再玩：

```text
NVMe
 ↓
GPUDirect Storage
 ↓
GPU
```

这时候你的项目就已经完全不是：

> “我会写一个 RMSNorm kernel。”

而变成：

> **“我做了一个面向 LLM KV Cache 的 GPU—CPU—SSD 分层缓存系统，并实现 CUDA 量化/反量化、异步数据搬运、prefetch、eviction 和性能 benchmark。”**

这个项目同时能覆盖：

```text
CUDA
+
LLM
+
vLLM
+
KV Cache
+
GPU Memory
+
PCIe
+
NVMe
+
Storage
```

这就非常像我们说的那个交叉方向。

而且以你目前正在做的**单卡 Qwen + CUDA 算子学习环境**来说，这条路线是可以从很小的 demo 一层层做起来的，不要求一开始就有服务器集群。

---

## 14. 最后给你定位一下这个方向

如果把几个方向画在一起：

```text
                    AI Algorithm
                         │
                         │
                  Transformer
                         │
                         │
CUDA Kernel ─────── LLM Serving ───── Distributed System
     │                   │
     │                   │
     └──── GPU Memory ───┘
              │
              │
         KV Cache
              │
       ┌──────┴──────┐
       │             │
      DRAM          SSD
       │             │
       └──── Storage ┘
```

**KV Cache 正好位于这些技术的交叉中心。**

所以如果你以后看到岗位描述里同时出现：

> CUDA / GPU、LLM Inference、KV Cache、Memory、Offload、CXL、NVMe、RDMA、GPUDirect、Tiered Storage

那基本就是我们现在说的这个方向。

而且它和“纯算子工程师”相比有一个明显区别：**纯算子追求某个 kernel 快 20%；这个方向追求的是让整个 `计算—显存—内存—SSD` 数据链路更高效。**

对你来说，下一阶段最值得真正搞明白的一条知识链，我会排成：

**`KV Cache → PagedAttention → pinned memory → cudaMemcpyAsync → CUDA stream → CPU offload → prefetch/eviction → NVMe → GDS`**。

一旦这条链串起来，你之前问的“**算子为什么能和存储公司接上**”基本就彻底打通了。

---

## 问题二：好的，把这条知识链串起来

可以。把这条链真正串起来以后，你会发现它不是 8 个零散名词，而是一条非常自然的“**问题逼出下一个技术**”的路线：

```text
KV Cache 太大
   ↓
不能粗暴地整块管理
   ↓
PagedAttention：把 KV 切成 page/block
   ↓
block 可以独立搬运
   ↓
GPU 放不下 → CPU Offload
   ↓
GPU ↔ CPU 搬运太慢 → Pinned Memory
   ↓
不能让 GPU 等搬运 → cudaMemcpyAsync + CUDA Stream
   ↓
按需加载还是会卡 → Prefetch
   ↓
GPU 满了要决定踢谁 → Eviction
   ↓
CPU 也装不下 → NVMe SSD
   ↓
SSD → CPU → GPU 中转又有开销
   ↓
GPUDirect Storage
   ↓
最终变成 GPU + AI Infra + Storage 的软硬件协同
```

我从头给你建立这个“因果链”。

---

# 一、起点：为什么突然有了 KV Cache 这个存储问题？

Transformer 解码时，每层 Attention 都会产生：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

假设已经生成：

```text
我
今天
准备
学习
CUDA
```

历史 token 的 K、V 已经算过了。

下一步生成新 token 时，没有必要重新计算：

```text
我      → K1,V1
今天    → K2,V2
准备    → K3,V3
学习    → K4,V4
CUDA    → K5,V5
```

于是把它们保存起来：

```text
┌────────────────────┐
│      KV Cache      │
├────────────────────┤
│ K1 V1              │
│ K2 V2              │
│ K3 V3              │
│ K4 V4              │
│ K5 V5              │
└────────────────────┘
```

下一个 token 只算新的：

```text
K6,V6
```

然后 Attention：

```text
Q6
 ↓
K1 K2 K3 K4 K5 K6
V1 V2 V3 V4 V5 V6
```

---

# 二、第一个矛盾：KV Cache 越来越大

KV Cache 大小可以粗略理解成：

\[
\text{KV大小}
\approx
2\times
L\times
H_{kv}\times
D\times
N\times
\text{bytes}
\]

其中：

```text
2        → K + V
L        → Transformer层数
Hkv      → KV head数量
D        → 每个head维度
N        → token数量
bytes    → FP16就是2 bytes
```

最关键的是：

\[
KV Cache \propto N
\]

上下文越长，KV 越大。

而且用户越多：

```text
用户A → KV Cache A
用户B → KV Cache B
用户C → KV Cache C
用户D → KV Cache D
```

都往 GPU 显存里塞。

于是问题出现：

> GPU 算力可能还有，但是显存先满了。

这就是整个知识链的起点。

---

# 三、第二个问题：怎么管理这么多 KV？

最原始的想法是：

```text
用户A：
申请一大片连续显存

用户B：
再申请一大片

用户C：
再申请一大片
```

问题是每个用户长度不同：

```text
A：10000 tokens
B：800 tokens
C：50000 tokens
D：2000 tokens
```

而且：

```text
A结束
C继续
B增长
D结束
```

于是 GPU 内存很容易变成：

```text
██████........████████....███.....████

█ = 已使用
. = 空闲
```

虽然总空闲空间可能很多，但是到处分散。

这就是：

> **内存碎片 fragmentation**

---

# 四、于是出现 PagedAttention

这里第一次和操作系统的思想接上了。

你学操作系统的话应该见过：

```text
Virtual Memory
Page
Page Table
```

PagedAttention 很像这个思想：

> 不要求一个请求的 KV Cache 在物理显存里连续。

把 KV Cache 切成很多小块：

```text
逻辑上的 KV Cache

token
1────16   → Block 0
17───32   → Block 1
33───48   → Block 2
49───64   → Block 3
```

但是 GPU 物理内存里可以是：

```text
GPU Memory

Physical Block 0 → 用户B
Physical Block 1 → 用户A block0
Physical Block 2 → 用户C
Physical Block 3 → 用户A block2
Physical Block 4 → 用户D
Physical Block 5 → 用户A block1
```

用户 A 看起来是：

```text
A:

Logical block 0
Logical block 1
Logical block 2
```

实际上：

```text
block 0 → Physical 1
block 1 → Physical 5
block 2 → Physical 3
```

所以需要一个：

```text
Block Table

logical block
      ↓
physical block
```

这正是 PagedAttention 的核心思想之一：把请求的 KV Cache 划分为固定 token 数的 KV blocks，并允许它们位于非连续物理内存中，从而实现按需分配、降低碎片问题。([docs.vllm.ai](https://docs.vllm.ai/en/v0.5.3.post1/automatic_prefix_caching/details.html?utm_source=chatgpt.com))

---

# 五、这里发生了一个非常关键的变化

在没有分页以前：

```text
KV Cache = 一坨巨大的 Tensor
```

分页以后：

```text
KV Cache
 ↓
Block 0
Block 1
Block 2
Block 3
...
```

突然之间：

> **KV Cache 从一个 Tensor，变成了很多可以独立管理的“数据块”。**

这一步非常重要。

因为接下来我们就可以：

```text
Block 1 → GPU

Block 2 → GPU

Block 3 → CPU

Block 4 → SSD
```

所以：

> **PagedAttention 是 LLM 算子世界通往“存储系统世界”的一座桥。**

---

# 六、第三个问题：GPU 放不下怎么办？

现在有 block 了。

假设 GPU 只能放：

```text
1000 blocks
```

但现在有：

```text
5000 blocks
```

最自然的想法：

```text
热数据
 ↓
GPU

暂时不用的数据
 ↓
CPU RAM
```

于是产生：

# CPU Offload

系统变成：

```text
          Attention
              │
              ↓
       ┌──────────────┐
       │ GPU KV Cache │
       │   HOT        │
       └──────┬───────┘
              │
             PCIe
              │
       ┌──────▼───────┐
       │ CPU KV Cache │
       │   WARM       │
       └──────────────┘
```

当前 vLLM 就已经有正式的 KV offloading 机制：完成的 KV blocks 可以被 offload 到 CPU 等更慢但容量更大的层级，需要时重新 promotion 回 GPU。([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/?utm_source=chatgpt.com))

注意这里开始真正进入：

> **AI Infra**

因为这不再是 Transformer 数学公式问题，而是：

```text
数据在哪里？
什么时候搬？
搬多少？
谁管理？
```

---

# 七、第四个问题：CPU → GPU 怎么搬？

最简单：

```cpp
cudaMemcpy(...)
```

例如：

```text
CPU KV Block
    ↓
cudaMemcpy
    ↓
GPU KV Block
```

但是有个问题。

普通 CPU 内存，例如：

```cpp
malloc(...)
```

操作系统可以把它对应的物理页移动、换出。

GPU DMA Engine 如果想直接搬：

```text
CPU RAM
  ↓
PCIe
  ↓
GPU
```

希望 CPU 这块内存：

> “你别乱跑，我正在搬。”

于是产生：

# Pinned Memory

也叫：

```text
Page-Locked Memory
```

它可以用类似：

```cpp
cudaMallocHost(...)
```

来申请。

直观理解：

```text
普通内存

OS：
这个 page 我想移动就移动


Pinned Memory

OS：
这个 page 被钉住了
不要乱动
```

Pinned memory 通常能获得更好的 Host↔GPU 传输性能，并且对真正的异步 Host↔Device 传输非常关键。NVIDIA 当前 CUDA 文档也明确说明，涉及 CPU 内存的异步传输需要 page-locked/pinned host buffer，否则不能获得预期的 copy/compute overlap。([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html?utm_source=chatgpt.com))

所以现在：

```text
GPU
 ↑
 │ DMA
 │
Pinned CPU Memory
```

这就是为什么你研究 KV Offload 时会突然碰到：

```text
cudaMallocHost
cudaHostAlloc
cudaHostRegister
```

---

# 八、第五个问题：搬数据的时候 GPU 难道干等？

假设：

```text
Attention计算：2 ms

CPU→GPU搬KV：3 ms
```

最笨：

```text
时间 →

搬数据：
██████

计算：
      ████

总时间 ≈ 5 ms
```

这很亏。

我们真正希望：

```text
时间 →

计算 Block A：
████████

同时

搬 Block B：
   ████████
```

于是：

```text
计算 A
+
搬运 B
```

重叠。

这就是：

# cudaMemcpyAsync + CUDA Stream

---

# 九、CUDA Stream 你可以先理解成“GPU任务队列”

比如：

```text
Stream 0:
Kernel A → Kernel B → Kernel C
```

同一个 stream 内：

```text
严格按顺序
```

但是可以有：

```text
Stream Compute

Kernel A
████████████████


Stream Copy

Memcpy B
    ████████████
```

不同 stream 有机会并行。

NVIDIA 对 stream 的定义本质就是一条有序的工作队列；`cudaMemcpyAsync()` 可以把传输任务放入指定 stream，而不同 stream 中的传输和 kernel 在硬件支持的情况下可以实现重叠。([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html?utm_source=chatgpt.com))

于是：

```cpp
cudaMemcpyAsync(
    gpu_block_B,
    cpu_block_B,
    size,
    cudaMemcpyHostToDevice,
    copy_stream);
```

与此同时：

```cpp
attention_kernel<<<..., compute_stream>>>(
    gpu_block_A);
```

就有可能形成：

```text
GPU

Compute Engine:
A ████████████

Copy Engine:
    B ████████████
```

---

# 十、这一步之后，你已经不只是“CUDA 算子工程师”了

因为单纯 CUDA kernel 学的是：

```text
thread
block
warp
shared memory
global memory
```

现在突然多了：

```text
GPU memory
CPU pinned memory
PCIe
DMA engine
CUDA stream
CUDA event
copy/compute overlap
```

所以技术栈从：

```text
Kernel
```

扩大到了：

```text
Kernel
  +
Runtime
  +
Memory System
```

这就是你往 AI Infra 走的一步。

---

# 十一、第六个问题：Async 还不够

假设 Attention 突然需要：

```text
KV Block 100
```

但它在 CPU。

系统才发现：

```text
啊！Block 100 不在 GPU！
```

然后开始：

```text
CPU
 ↓
GPU
```

即使是 Async，也晚了。

因为 Attention：

```text
我要 Block100
```

但是：

```text
Block100：
我还在 PCIe 上……
```

GPU：

```text
等
等
等
```

这叫：

> cache miss 带来的 stall。

---

# 十二、于是产生 Prefetch

核心思想非常简单：

> **不要等数据要用了才搬，提前搬。**

例如现在正在处理：

```text
Block 100
```

系统预测：

```text
马上需要：
101
102
103
```

那么：

```text
计算 100
  │
  ├───────────────┐
  ↓               ↓
GPU Compute    CPU→GPU
Block100       Block101
```

然后：

```text
计算101
  │
  ├───────────────┐
  ↓               ↓
GPU            搬102
```

形成流水线：

```text
时间 ─────────────────────────────→

Compute:
[100][101][102][103][104]

Transfer:
    [101][102][103][104][105]
```

理想情况：

> 数据搬运时间被计算时间“藏起来”。

英文你以后会经常看到：

> **Hide data-transfer latency behind computation.**

这句话非常重要。

---

# 十三、Prefetch 其实已经有“算法味”了

问题来了：

> 提前搬谁？

这就不是 CUDA API 可以告诉你的了。

系统需要预测：

```text
哪些 block 马上会用？
```

可能根据：

```text
当前请求
Attention访问模式
历史访问
Prefix
Scheduler
优先级
```

决定。

如果猜错：

```text
提前搬了100个block
结果都没用
```

反而：

```text
浪费PCIe带宽
+
浪费GPU显存
```

所以你前面看到所谓：

> **“智能分层存储”**

“智能”就可以体现在这里。

---

# 十四、第七个问题：GPU 满了，谁出去？

假设 GPU 有：

```text
Block A
Block B
Block C
Block D
```

现在要把：

```text
Block E
```

搬进来。

但是满了。

必须：

```text
踢出去一个
```

那么踢谁？

于是进入传统缓存系统最经典的问题：

# Eviction

例如：

```text
LRU
Least Recently Used
```

即：

> 最久没使用的先出去。

例如：

```text
A：刚用
B：刚用
C：20秒没用
D：2秒没用
```

那么：

```text
C
 ↓
GPU → CPU
```

当然实际 LLM 系统可以比 LRU 复杂得多。vLLM 的 prefix-cache 设计中就存在 block 级的缓存和 eviction 逻辑；当前 KV cache manager 本身也已经是独立的 block 管理系统。([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/?utm_source=chatgpt.com))

到了这里，你应该发现：

> 这越来越像 Redis / 数据库 Buffer Pool / CPU Cache 了。

没错。

---

# 十五、现在我们的系统已经长这样

```text
                 LLM Scheduler
                      │
             ┌────────┴─────────┐
             │                  │
          Prefetch           Eviction
             │                  │
             └────────┬─────────┘
                      ↓

                 Block Manager
                      │

       ┌──────────────┴─────────────┐
       ↓                            ↓
GPU KV Blocks                 CPU KV Blocks
     HOT                           WARM

       ↑                            ↓
       └── cudaMemcpyAsync / DMA ──┘
```

现在已经同时包含：

```text
LLM
CUDA
Cache Algorithm
Memory Management
Runtime
```

---

# 十六、第八个问题：CPU RAM 也不是无限的

服务器可能：

```text
GPU HBM：80 GB
CPU RAM：512 GB
```

看起来 CPU 很大。

但是如果：

```text
几千个请求
+
超长上下文
```

CPU 也会装不下。

那怎么办？

还有一个更大的仓库：

# NVMe SSD

于是：

```text
             HOT
              ↓

        GPU HBM
        80 GB
          ↑↓
        PCIe
          ↑↓
        CPU RAM
        512 GB
          ↑↓
        NVMe SSD
         4 TB

              ↓
             COLD
```

这就是真正意义上的：

# Tiered Storage

也就是：

```text
Tier 0：GPU

Tier 1：CPU DRAM

Tier 2：NVMe SSD
```

vLLM 当前的 KV offloading 设计已经明确支持 CPU primary tier 加可选 secondary tiers；其文档指出 secondary tier 与 GPU 的传输会经过 CPU primary tier。([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/?utm_source=chatgpt.com))

这句话特别值得你注意，因为它正好引出下一环。

---

# 十七、普通 NVMe Offload 的数据路径

如果 KV 在 SSD：

```text
SSD
 ↓
CPU RAM
 ↓
GPU
```

也就是：

```text
NVMe SSD
   │
   ↓
CPU Buffer
   │
   ↓
Pinned Buffer
   │
   ↓
PCIe
   │
   ↓
GPU HBM
```

这是经典：

> Storage → Host → Device

但这里存在 CPU bounce buffer。

---

# 十八、为什么这个中转可能成为问题？

想象搬家公司：

```text
仓库 A：SSD
仓库 B：CPU RAM
仓库 C：GPU
```

传统：

```text
SSD
 ↓
先卸到CPU仓库
 ↓
再重新装车
 ↓
送GPU
```

如果数据巨大：

```text
50 GB
100 GB
500 GB
```

中间搬运就是额外工作。

于是有人自然会问：

> 能不能 SSD 直接把数据送给 GPU？

这就来到最后一个环节。

---

# 十九、GPUDirect Storage

简称：

# GDS

普通路径：

```text
     SSD
      │
      ↓
   CPU RAM
      │
      ↓
   GPU HBM
```

GDS 希望建立：

```text
SSD ───────────→ GPU HBM
       DMA
```

尽量绕过 CPU bounce buffer。

NVIDIA 对 GDS 的定义正是：在 storage 和 GPU memory 之间建立直接 DMA 数据路径，从而避免数据必须通过 CPU bounce buffer 中转，并降低 CPU 占用及相关瓶颈。([docs.nvidia.com](https://docs.nvidia.com/gpudirect-storage/overview-guide/?utm_source=chatgpt.com))

注意：

> “绕过 CPU buffer”不等于“整个系统完全没有 CPU”。

CPU 仍然负责很多控制逻辑。

重点是：

```text
大块数据
```

不必：

```text
SSD → CPU buffer → GPU
```

这样来回倒。

---

# 二十、现在整条链终于串起来了

你现在重新看：

```text
KV Cache
 ↓
PagedAttention
 ↓
Pinned Memory
 ↓
cudaMemcpyAsync
 ↓
CUDA Stream
 ↓
CPU Offload
 ↓
Prefetch / Eviction
 ↓
NVMe
 ↓
GDS
```

每一个都不是突然冒出来的。

它们之间是严格因果关系：

| 技术 | 它解决的上一个问题 |
|---|---|
| KV Cache | 避免重复计算历史 K/V |
| PagedAttention | KV 太大、动态长度导致显存管理困难 |
| CPU Offload | GPU 装不下所有 KV |
| Pinned Memory | CPU↔GPU 高效 DMA |
| cudaMemcpyAsync | 搬数据时不要阻塞 |
| CUDA Stream | Copy 和 Compute 并行 |
| Prefetch | 不要等缺页才开始搬 |
| Eviction | GPU 满了决定谁出去 |
| NVMe | CPU 内存也不够 |
| GDS | 减少 SSD→CPU→GPU 中转 |

**这张表就是整个方向的知识地图。**

---

# 二十一、那 CUDA 算子到底在哪里？

现在再回到你最开始的问题。

Attention Kernel：

```text
Q
 │
 ↓
┌──────────────────────────┐
│   Paged Attention Kernel │
└─────────────┬────────────┘
              ↑
              │
       GPU KV Blocks
              ↑
              │
        DMA / Prefetch
              ↑
              │
        CPU / SSD
```

Kernel 最终还是需要：

```text
K
V
```

但这些 K/V：

> **什么时候到？在哪？什么 layout？什么 dtype？是否已经解压？**

全部是 Storage / Runtime 决定的。

所以真正高级的系统优化不是：

```text
Kernel                Storage
  ↓                       ↓
各玩各的               各玩各的
```

而是：

```text
              Attention Kernel
                    ↑
                    │
             KV Layout设计
                    ↑
                    │
       Quant / Dequant Kernel
                    ↑
                    │
           Prefetch / Eviction
                    ↑
                    │
          GPU ↔ CPU ↔ SSD
```

这就叫：

# Software-Hardware Co-design / System Co-design

或者你也可以简单理解成：

> **计算和存储协同优化。**

---

# 二十二、你现在应该怎么学？

以你目前 CUDA 还是初学阶段的情况，我不建议你直接上 GDS。

正确顺序是下面这一条。

1. **先彻底搞懂 KV Cache。** 自己用一个极简 Attention 写出“不带 KV Cache”和“带 KV Cache”的 decode，能看懂 `[batch, kv_head, seq, head_dim]` 这些维度。

2. **再理解 PagedAttention。** 暂时不要啃 CUDA kernel 源码，只理解 `page/block、block table、logical block、physical block`。把它想成操作系统虚拟内存。

3. **单独学习 CPU↔GPU 数据搬运。** 写一个 100 行左右 CUDA demo，对比 `malloc + cudaMemcpy`、`cudaMallocHost + cudaMemcpy`、`cudaMallocHost + cudaMemcpyAsync`。

4. **学习 CUDA Stream 和 Event。** 做两个 stream，一个负责 memcpy，一个负责简单 kernel，真正用 Nsight Systems 看见 `Memcpy HtoD` 和 Kernel 在时间轴上重叠。

5. **自己实现一个极简 CPU KV Offload。** 把 KV 切成 block，GPU 只能容纳比如 8 个 block，其余放 pinned CPU memory，需要的时候搬回来。

6. **加入 Prefetch + Eviction。** 先用最简单 LRU，然后做 double buffer，让“计算 block N”和“加载 block N+1”并行。

7. **再加入 SSD。** CPU 放不下的 block 写到 NVMe，形成 `GPU → CPU → SSD` 三层缓存。先用普通文件 I/O，不要碰 GDS。

8. **最后再研究 GDS。** 到这一步你自然会明白为什么需要它，而不是单纯背 `cuFile` API。

---

# 二十三、尤其建议你做这个实验

你的单卡环境其实很适合先做一个 **Mini KV Cache Manager**。

第一版甚至不用 Qwen：

```text
假装有 100 个 KV blocks

每个 block = 16 MB
```

GPU 只允许：

```text
8 blocks
```

CPU 保存剩下：

```text
92 blocks
```

然后模拟 Attention 按：

```text
0
1
2
3
...
```

访问。

第一阶段：

```text
需要 block 8
↓
cudaMemcpy
↓
等待
↓
kernel
```

测：

```text
总耗时
```

第二阶段：

```text
kernel(block 7)
       +
cudaMemcpyAsync(block 8)
```

再测。

你第一次真正看到：

```text
同步版本：

COPY ███
          KERNEL ██
                   COPY ███
                            KERNEL ██


异步流水：

COPY ███
    KERNEL ███
       COPY ███
           KERNEL ███
```

的时候，

**Pinned Memory → Async Copy → Stream → Prefetch → Offload**

这五个概念会一下子全串起来。

等这个实验做完，再把：

```text
fake KV block
```

替换成：

```text
Qwen 真正的 KV Cache block
```

整个项目就开始从“CUDA 教学 demo”变成真正的 **LLM Infra 项目**了。

你当前其实不用急着学 NVMe/GDS。**最近最关键的下一站应该是先把 `KV Cache → PagedAttention` 搞透，然后亲手做 `Pinned Memory + cudaMemcpyAsync + Stream`。**等这三块掌握以后，CPU Offload 几乎就是顺理成章的事情。
