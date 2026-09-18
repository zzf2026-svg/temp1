# 大模型算子（LLM Kernel）与 TIRX / Tensor Compiler 学习路线

## 学习目标

目标：建立从"大模型算子 → TensorIR/TIRX → CUDA Kernel →
GPU硬件执行"的完整认知。

整体链路：

    PyTorch模型
        ↓
    计算图 Graph
        ↓
    算子 Operator
        ↓
    TensorIR / TIRX
        ↓
    Schedule调度
        ↓
    GPU线程映射
        ↓
    CUDA Kernel
        ↓
    PTX/SASS
        ↓
    GPU硬件执行

------------------------------------------------------------------------

# 第一阶段：GPU基础与CUDA编程模型

## 目标

理解：

-   thread
-   block
-   warp
-   SM
-   GPU memory hierarchy
-   global memory / shared memory / register
-   CUDA kernel执行模型

这是理解TIRX、Triton、FlashAttention的基础。

## 推荐学习资料

### CUDA Programming Guide

https://docs.nvidia.com/cuda/cuda-c-programming-guide/

重点：

-   Thread hierarchy
-   Memory hierarchy
-   CUDA execution model

------------------------------------------------------------------------

# 第二阶段：深度学习编译器基础

## 1. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (2018)

论文：

https://arxiv.org/abs/1802.04799

核心内容：

TVM提出端到端深度学习编译框架：

    Deep Learning Model
            ↓
    Graph IR
            ↓
    Tensor Expression
            ↓
    TIR
            ↓
    CUDA Code

重点学习：

-   IR（Intermediate Representation）
-   Schedule思想
-   自动生成高性能Kernel

理解：

一个计算：

C=A×B

如何从：

    for i
     for j
      for k

变成：

    GPU block
        ↓
    thread
        ↓
    warp

------------------------------------------------------------------------

# 第三阶段：TensorIR / TIRX核心

## 2. TensorIR: An Abstraction for Automatic Tensorized Program Optimization (2022)

论文：

https://arxiv.org/abs/2207.04296

TensorIR是TVM现代Tensor编译的重要基础。citeturn0search0

官方文档：

https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html

核心思想：

把：

-   tensor computation
-   loop structure
-   memory access
-   hardware mapping

统一到一个IR中。

学习重点：

## TensorIR核心对象

### PrimFunc

表示一个底层Tensor程序。

### Block

TensorIR最重要概念：

    Block
     |
    计算区域
     |
    输入Buffer
     |
    输出Buffer
     |
    循环变量

### Schedule

修改程序结构：

例如：

原始：

    for i
     for j
      for k

优化：

    block
     thread
     vector
     shared memory

重点API：

-   split
-   fuse
-   reorder
-   bind
-   cache_read
-   cache_write
-   tensorize

------------------------------------------------------------------------

# 第四阶段：Triton理解现代Kernel DSL

## 3. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (2021)

论文：

https://arxiv.org/abs/2103.15579

Triton目标：

降低CUDA Kernel开发难度。

CUDA:

    threadIdx
    blockIdx
    warp

Triton:

    program_id
    load
    store
    tile

重点理解：

-   tile计算
-   memory coalescing
-   vectorized load/store
-   GPU mapping

------------------------------------------------------------------------

# 第五阶段：LLM核心算子论文

------------------------------------------------------------------------

# 4. FlashAttention (2022)

论文：

https://arxiv.org/abs/2205.14135

核心思想：

Attention性能瓶颈不是计算，而是显存访问。

普通Attention：

    QK^T
     ↓
    softmax
     ↓
    ×V

产生巨大中间矩阵。

FlashAttention：

使用tiling：

    HBM
     ↓
    SRAM
     ↓
    Compute
     ↓
    Next Tile

重点对应TIR：

-   cache
-   tile
-   shared memory
-   fusion

------------------------------------------------------------------------

# 5. FlashAttention-2 (2023)

论文：

https://arxiv.org/abs/2307.08691

重点：

-   更好的GPU并行设计
-   更高Tensor Core利用率
-   warp级优化

学习：

-   warp scheduling
-   thread mapping
-   GPU occupancy

------------------------------------------------------------------------

# 第六阶段：LLM Kernel工程实践

## 6. Liger Kernel (2024)

论文：

https://arxiv.org/abs/2410.10989

适合学习：

-   RMSNorm Kernel
-   RoPE Kernel
-   SwiGLU Kernel
-   Transformer训练Kernel优化

------------------------------------------------------------------------

## 7. TensorRT-LLM / FasterTransformer

TensorRT-LLM:

https://github.com/NVIDIA/TensorRT-LLM

学习：

-   fused operator
-   KV Cache
-   attention优化
-   LLM inference pipeline

------------------------------------------------------------------------

# 第七阶段：自动优化与Schedule搜索

## 8. Ansor: Generating High-Performance Tensor Programs for Deep Learning (2020)

论文：

https://arxiv.org/abs/2006.06762

核心：

自动搜索最优schedule。

流程：

    生成schedule空间

    ↓

    benchmark

    ↓

    cost model

    ↓

    选择最快kernel

------------------------------------------------------------------------

## 9. MetaSchedule (2022)

论文：

https://arxiv.org/abs/2205.13603

重点：

TensorIR时代的自动调度系统。

学习：

-   search space
-   cost model
-   schedule optimization

------------------------------------------------------------------------

# 第八阶段：现代LLM Compiler方向

## 10. Relax: Composable Abstractions for End-to-End Dynamic Machine Learning (2023)

论文：

https://arxiv.org/abs/2311.02103

关注：

-   动态shape
-   LLM部署
-   Graph IR + Tensor IR融合

------------------------------------------------------------------------

## 11. FlexAttention

项目：

https://pytorch.org/blog/flexattention/

目标：

通过高级Attention描述自动生成高性能Kernel。

------------------------------------------------------------------------

# 推荐阅读顺序（适合当前TIRX学习）

## 第1个月

    CUDA Programming Model

    ↓

    TVM 2018

    ↓

    TensorIR

    ↓

    Triton

目标：

看懂：

    sch.bind(threadIdx.x)
    sch.cache_read()

为什么存在。

------------------------------------------------------------------------

## 第2个月

    FlashAttention

    ↓

    FlashAttention-2

    ↓

    Liger Kernel

    ↓

    TensorRT-LLM

目标：

能够理解：

-   Attention Kernel
-   RMSNorm Kernel
-   RoPE Kernel

------------------------------------------------------------------------

## 第3个月

进入TIRX源码：

顺序：

    TensorIR Block

    ↓

    Schedule Primitive

    ↓

    CUDA Lowering

    ↓

    Tensor Core Tensorize

    ↓

    MetaSchedule

------------------------------------------------------------------------

# 最终能力目标

能够完成：

    PyTorch算子

    ↓

    数学表达

    ↓

    TIR/TIRX

    ↓

    Schedule优化

    ↓

    CUDA Kernel

    ↓

    Benchmark

    ↓

    性能分析

推荐练习路线：

    Matmul

    ↓

    RMSNorm

    ↓

    Softmax

    ↓

    RoPE

    ↓

    FlashAttention

    ↓

    PagedAttention

每个算子同时实现：

-   PyTorch版本
-   CUDA版本
-   Triton版本
-   TIRX版本

这是进入大模型Kernel优化岗位最有效的路线。
