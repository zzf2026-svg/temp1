# GPU Kernel + ML Compiler + LLM Systems 学习论文与网站路线

如果你的目标是往 **GPU kernel + ML compiler + LLM systems** 这条线走，我建议不要“广撒网”读论文，而是按一个固定顺序读：先建立性能直觉，再学 kernel，再学 compiler，最后碰 megakernel 和最新工作。

---

## 第一批：现在就可以开始读的 4 篇

### 1. Roofline Model

**Samuel Williams et al., _Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures_, 2009**

这是我最建议你第一篇读的。

它会让你建立一个非常核心的概念：

```text
程序性能
   ↓
到底受什么限制？

计算能力不够？
还是内存带宽不够？
```

也就是：

```text
Compute Bound
vs
Memory Bound
```

以后你看到：

```text
Arithmetic Intensity
HBM bandwidth
FLOPS
memory-bound kernel
compute-bound kernel
```

就不会懵。

Roofline 本身就是为了帮助开发者直观判断性能瓶颈而设计的。

**论文链接：**
- [Roofline 原文页面](https://escholarship.org/uc/item/5tz795vq)

读这篇的时候，你只需要真正搞懂：

```text
Arithmetic Intensity = FLOPs / Bytes

Performance ≈ min(
    Peak Compute,
    Memory Bandwidth × Arithmetic Intensity
)
```

**先别纠结所有公式。**

---

### 2. FlashAttention

**Tri Dao et al., _FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness_, NeurIPS 2022**

这篇论文是 GPU 性能工程非常好的入门教材。

它最重要的地方不是 Attention，而是告诉你：

> **少算 FLOPs 不代表程序一定更快，数据搬运也可能才是真正的瓶颈。**

FlashAttention 的核心思路就是利用 tiling，让数据更多停留在 GPU 片上 SRAM，减少 HBM 与片上存储之间的数据传输。

**论文链接：**
- [FlashAttention 论文](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)

你第一次读的时候重点只看：

```text
普通 attention 为什么慢？

HBM 是什么？
SRAM 是什么？

为什么 tiling 有用？

为什么不能直接存 N×N attention matrix？

online softmax 是怎么让 tiling 成立的？
```

**数学证明第一次可以跳。**

这篇如果真的吃透，你的 GPU intuition 会提升非常多。

---

### 3. TVM

**Tianqi Chen et al., _TVM: An Automated End-to-End Optimizing Compiler for Deep Learning_, OSDI 2018**

这是你进入 ML compiler 世界的第一篇。

TVM 的核心问题是：

> 同一个 tensor computation，怎么自动映射到 CPU、GPU 和各种 accelerator，同时获得高性能？

它把 graph optimization、operator optimization、hardware mapping 和自动搜索放进一个编译系统里。

**论文链接：**
- [TVM OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/chen)

第一次读的时候，不要追 implementation。

搞懂这几个概念：

```text
Computation

和

Schedule
```

为什么应该分开。

例如数学上都是：

```python
C = A @ B
```

但实现可以变成：

```text
tile
reorder
vectorize
parallel
cache
thread binding
```

数学语义不变，性能却完全不同。

**这是整个 TVM / TensorIR 世界的核心思想之一。**

---

### 4. Triton

**Philippe Tillet, H. T. Kung, David Cox, _Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations_, 2019**

这篇是连接：

```text
CUDA programmer
        ↕
compiler
```

非常好的一座桥。

Triton 的核心思想之一，就是让程序员更多地思考 **tile / block of data**，而不是手动管理每一个 CUDA thread。OpenAI 对 Triton 的介绍也明确把它定位为用于高性能 DNN kernel 的语言和编译器。

**论文 / 项目入口：**
- [OpenAI Triton 介绍与论文入口](https://openai.com/index/triton/)

如果这篇第一次看觉得抽象，**完全正常**。

论文看个大概后立刻写 Triton，比硬读三遍有用。

---

# 第二批：CUDA 写了一阵之后再读

### 5. FlashAttention-2

这时候开始研究：

```text
怎么划分 thread block？
怎么划分 warp？
occupancy 为什么重要？
为什么 work partitioning 会决定性能？
```

FlashAttention-2 就是在 FlashAttention 基础上进一步优化工作划分、occupancy 和 warp 间通信。

**论文链接：**
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)

你会开始真正理解：

```text
algorithm 快
≠
GPU implementation 快
```

---

### 6. Ansor

**_Ansor: Generating High-Performance Tensor Programs for Deep Learning_, OSDI 2020**

这篇回答一个很有意思的问题：

> schedule 参数那么多，人能不能不手调，让机器自己搜索？

Ansor 使用层次化搜索空间、evolutionary search 和 learned cost model 来寻找高性能 tensor program。

**论文链接：**
- [Ansor 论文](https://arxiv.org/abs/2006.06762)

你可以把它理解成：

```text
matmul

可能存在：

tile = ?
unroll = ?
vector width = ?
thread binding = ?
cache strategy = ?

                 ↓

自动搜索
                 ↓

高性能 program
```

这也是理解后来 MetaSchedule 的铺垫。

---

### 7. TensorIR

这篇就非常重要了，因为 **Hongyi Jin 本人就是作者之一**。

**_TensorIR: An Abstraction for Automatic Tensorized Program Optimization_**

TensorIR 的目标是设计一种更适合 tensor program 和硬件 tensor primitive 的 compiler abstraction。

**论文链接：**
- [TensorIR 论文](https://arxiv.org/abs/2207.04296)

到这里，你要开始认真理解：

```text
IR 是什么？

为什么不能直接优化 Python/CUDA source？

block 是什么？

loop transformation 是什么？

tensorization 又是什么？
```

你之前看到 `jinhongyii` 早期那些：

```text
Fuse
Split
Reorder
MetaSchedule
```

PR，就会逐渐开始串起来。

---

### 8. MetaSchedule

这篇也直接有 Hongyi。

**_Tensor Program Optimization with Probabilistic Programs_, NeurIPS 2022**

它提出 MetaSchedule，用 probabilistic programming 的方式组织 tensor program optimization 的搜索空间。

**论文链接：**
- [MetaSchedule 论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e894eafae43e68b4c8dfdacf742bcbf3-Abstract-Conference.html)

它比较难。

如果你还没写过 TensorIR schedule，**先别看**。

不然非常容易变成：

> 每个英文单词都认识，连起来不知道作者在干什么。

---

# 第三批：等你已经比较强之后

到那时候再看 Hongyi 最近这种东西。

### Event Tensor

**Hongyi Jin et al., _Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel_, MLSys 2026**

这是非常新的工作，也是 Hongyi 第一作者。

它针对现代 LLM inference 中 kernel launch overhead、粗粒度同步、dynamic shape 和 data-dependent computation，设计了用于 dynamic megakernel 的 Event Tensor abstraction。

**论文链接：**
- [Event Tensor — MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html)

**现在不要拿它当入门论文。**

你最终应该能走到这里：

```text
Roofline
   ↓
FlashAttention
   ↓
CUDA optimization
   ↓
Triton
   ↓
TVM
   ↓
TensorIR
   ↓
Ansor / MetaSchedule
   ↓
Megakernel
   ↓
Event Tensor
```

---

# 网站方面，我最建议你长期放在书签栏里的

## ① NVIDIA CUDA Programming Guide

这应该成为你的**字典**。

当前官方 Programming Guide 从 CUDA programming model 开始，一路覆盖 CUDA C++、SIMT kernel、tile kernel、async execution，后面还有 asynchronous barriers、pipelines、async copy、memory model 等高级内容。

**网站链接：**
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)

但是千万不要：

> 从第一页看到最后一页再开始写 CUDA。

应该是：

```text
今天写 reduction
    ↓
遇到 warp
    ↓
查 warp

今天写 shared memory
    ↓
遇到 bank conflict
    ↓
查 bank conflict
```

**把它当 reference。**

---

## ② CUDA Best Practices Guide

Programming Guide 更像：

> CUDA 是什么、功能怎么用。

Best Practices 更像：

> **怎么写快。**

它专门讨论 memory、parallel execution、instruction efficiency 等性能优化问题。

**网站链接：**
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

以后你的顺序应该经常是：

```text
写 kernel
   ↓
benchmark
   ↓
发现慢
   ↓
Best Practices
   ↓
改
   ↓
benchmark
```

---

## ③ Triton 官方 Tutorials

这个可能是**最适合你现在直接动手的网站之一**。

官方本身就建议按顺序阅读，而且目前教程从：

```text
Vector Addition
Fused Softmax
Matrix Multiplication
LayerNorm
Fused Attention
Group GEMM
Persistent Matmul
Block Scaled Matmul
```

一路往上。

**网站链接：**
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

我建议你：

> 每篇教程都不要只运行。

而要删掉以后重新写。

---

## ④ Machine Learning Compilation

**这个特别推荐你。**

是 Tianqi Chen 的 MLC 教材，而且有中文。

课程从 tensor program abstraction、TensorIR 一直到自动程序优化、GPU hardware acceleration 和 ML framework integration。

**网站链接：**
- [MLC 英文版](https://book.mlc.ai/)
- [MLC 中文版](https://book-zh.mlc.ai/chapter_introduction/index.html)

如果你中文阅读快，我建议：

**第一遍中文，第二遍关键章节英文。**

这个比你现在直接啃 compiler textbook 实用很多。

---

## ⑤ Apache TVM TensorIR 文档

等你开始 MLC 以后使用。

官方 TensorIR tutorial 会直接教你怎样用 TVMScript 创建 TensorIR function。

**网站链接：**
- [TensorIR Tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_creation.html)
- [TensorIR 文档主页](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html)
- [MetaSchedule Tutorial](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)

你以后应该做到看到：

```python
@T.prim_func
def matmul(...):
```

能在脑中大概想象：

```text
loop
block
buffer
thread binding
memory scope
```

到底表示什么。

---

## ⑥ Nsight Compute

等你已经会写几个 CUDA kernel 后，**越早学越好**。

Nsight Compute 是 NVIDIA 的 CUDA kernel profiler，可以看到详细的 hardware performance metrics。

**网站链接：**
- [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)

以后不要猜：

> “这个 kernel 应该是 shared memory 太慢。”

应该打开 profiler 看。

这是从：

```text
凭感觉优化
```

变成：

```text
performance engineering
```

的重要一步。

---

# 如果是我带你，我会让你前两个月只看这些

```text
网站：

CUDA Programming Guide
        +
CUDA Best Practices

        ↓

Triton Tutorials

        ↓

MLC Book


论文：

Roofline
   ↓
FlashAttention
   ↓
Triton
   ↓
TVM
```

**就够了。**

先别堆几十篇 paper。

最危险的事情就是：

```text
收藏：
137 篇

读过：
27 篇

真正实现：
0
```

---

# 一篇论文应该怎么读

特别是 systems paper，不要像数学教材那样逐字读。

第一遍只回答四个问题：

```text
1. 它解决什么问题？

2. 为什么以前的方法不够好？

3. 作者最核心的新想法是什么？

4. 实验有没有证明这个想法？
```

第二遍才看：

```text
algorithm
system design
implementation
benchmark
```

第三遍最重要：

> **自己实现其中一个小东西。**

比如 FlashAttention 不需要一上来复现完整 FlashAttention。

你先自己实现：

```text
naive softmax
↓
stable softmax
↓
online softmax
↓
tiled softmax
```

这时候你重新读 FlashAttention，感觉会完全不一样。

---

# 今天开始的三个入口

如果只让我给你**今天开始的三个入口**，我会选：

1. **Roofline**
2. **CUDA Programming Guide 的 Programming Model**
3. **Triton 的 Vector Add / Softmax 教程**

等你把这几个真正做过，而不是单纯“看过”，再进入 FlashAttention 和 TVM，会顺很多。

---

# 推荐阅读顺序总览

```text
阶段 1：性能直觉
├── Roofline
└── CUDA Programming Model

阶段 2：GPU Kernel
├── CUDA Best Practices
├── Triton Tutorials
├── FlashAttention
└── FlashAttention-2

阶段 3：ML Compiler
├── TVM
├── MLC Book
├── TensorIR
├── Ansor
└── MetaSchedule

阶段 4：前沿 GPU / LLM Systems
├── Megakernel
├── TIRx / TVM 最新 PR
└── Event Tensor
```

核心原则：

> **30% 阅读，70% 实现、benchmark、profiling 和 debug。**

不要追求“看过很多”，要追求“能解释、能实现、能测量、能优化”。
