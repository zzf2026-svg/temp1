可以把 TIRx 理解成一个“专门面向 GPU Kernel 的小型编程语言 + 编译器系统”。

它不是 PyTorch 那种“调用算子”，也不只是 Triton 那种“换一种方式写 kernel”。TIRx 更强调：**把 GPU 硬件本身暴露给程序员，同时又保留一层可编译、可变换的结构化 IR。** 官方目前把它作为 Apache TVM 的下一代 kernel-level compiler 体系。([tvm.apache.org](https://tvm.apache.org/docs/tirx/overview.html?utm_source=chatgpt.com))

## 1. 先建立一个最简单的脑图

你可以先不要想 TVM 那一大堆代码，只记这一条链：

```text
你写 Python TIRx
        ↓
解析成 IR
        ↓
识别 thread / buffer / layout / tile op
        ↓
Tile Primitive Dispatch
        ↓
把高级操作降成普通循环、地址计算、CUDA intrinsic/PTX
        ↓
各种 compiler pass
        ↓
拆成 CPU host launcher + GPU device kernel
        ↓
生成 CUDA C / PTX
        ↓
编译
        ↓
GPU 执行
```

真正理解 TIRx，80% 就是在理解这条流水线。官方 lowering pipeline 也是这个结构：`LowerTIRx` 先处理 tile primitive 和 layout，然后做 buffer flatten、vectorize、unroll、CSE、类型合法化等，之后 `SplitHostDevice` 把一个函数拆成 host 和 device 两部分，最后交给 CUDA codegen。([tvm.apache.org](https://tvm.apache.org/docs/tirx/arch/lowering_pipeline.html))

---

# 2. TIRx 为什么不能直接看成 Python？

例如你写一个很简单的 kernel，概念上类似：

```python
@Tx.prim_func
def add(A, B, C):
    Tx.device_entry()

    tx = Tx.thread_id([256])

    C[tx] = A[tx] + B[tx]
```

你看起来是在写 Python。

但编译器真正关心的不是 Python 本身，而是：

```text
Function
 ├── parameters
 │    ├── Buffer A
 │    ├── Buffer B
 │    └── Buffer C
 │
 └── body
      ├── DeviceEntry
      ├── ThreadId(tx, extent=256)
      └── Store
           ├── buffer = C
           ├── index = tx
           └── value =
                Add(
                    Load(A, tx),
                    Load(B, tx)
                )
```

这棵树就是 **IR——Intermediate Representation，中间表示**。

实际 TVMScript 有自己的 parser/IR builder。Python AST 被转换成 TVM/TIRx 的 IR，而不是把它当普通 Python kernel 执行。([tvm.apache.org](https://tvm.apache.org/docs/arch/tvmscript.html?utm_source=chatgpt.com))

所以你以后看 TIRx 源码时，要形成这个意识：

> Python 只是“用户输入格式”，真正的主角是 IR。

---

# 3. TIRx 最核心的三个设计

这三个东西理解了，你基本就理解了为什么要造 TIRx。

## 第一层：Execution Scope——谁执行

CUDA 里面有：

```text
GPU
 └── CTA / block
      ├── warp
      │    ├── thread
      │    ├── thread
      │    └── ...
      └── warp
```

一个操作可能是：

```text
1 个 thread 做
32 个 thread（warp）一起做
128 个 thread（warpgroup）一起做
整个 CTA 一起做
```

TIRx 不希望编译器猜。

所以会显式描述：

```text
thread scope
warp scope
warpgroup scope
CTA scope
```

为什么重要？

因为：

```text
copy()
```

如果一个线程执行，可能是普通 load/store。

如果一个 warp 执行，可能适合 `ldmatrix`。

如果一个指定线程负责发起异步搬运，可能使用 TMA。

所以：

```text
谁在执行
    ↓
影响
    ↓
选择什么硬件指令
```

这就是 execution scope。([tvm.apache.org](https://tvm.apache.org/docs/tirx/overview.html?utm_source=chatgpt.com))

---

# 4. 第二层：Layout——数据到底怎么分布

这是 TIRx 非常核心，同时也是以后最容易把你绕晕的地方。

假设有：

```text
A[128][128]
```

逻辑上它只是一个矩阵。

但到了 GPU 上，可能实际上是：

```text
global memory
    ↓
shared memory
    ↓
warp 0 负责一部分
warp 1 负责一部分
    ↓
lane 0 拿 element 0, 32...
lane 1 拿 element 1, 33...
    ↓
每个 thread 又放几个元素到 register
```

所以：

```text
逻辑矩阵

A[i][j]
```

和：

```text
这个元素实际在哪
```

完全不是一回事。

TIRx 的 layout 本质就是一个映射：

```text
logical coordinate

(i, j)

       ↓ Layout

physical coordinate

memory offset
thread id
lane id
register id
...
```

官方 TIRx 的 layout 比传统 `shape + stride` 更进一步，它的轴带有硬件语义，可以描述 memory、thread、device 等资源。([tvm.apache.org](https://tvm.apache.org/docs/tirx/layout.html?utm_source=chatgpt.com))

你目前不用急着理解什么：

```text
S
R
O
TLane
TCol
tid_in_wg
```

先把它想成：

> Layout = “矩阵中的这个元素，最终归哪个线程、哪个寄存器、哪个内存地址负责”。

就够了。

---

# 5. 第三层：Tile Primitive Dispatch——TIRx 最漂亮的一层

比如用户写：

```python
Tx.tile.copy(B_shared, A_global)
```

TIRx 不马上规定：

```text
必须用哪条 CUDA 指令。
```

它先在 IR 里面保留一个：

```text
TilePrimitiveCall(
    op = "copy",
    src = A,
    dst = B
)
```

然后编译阶段再问：

```text
目标 GPU 是什么？
        +
src 是 global 还是 shared？
        +
dst 是 shared 还是 register？
        +
layout 是什么？
        +
谁来执行？
```

之后决定实现。

概念上：

```text
                   copy
                     │
         ┌───────────┼─────────────┐
         ↓           ↓             ↓
普通 load/store    ldmatrix      async copy
                                   │
                                   ↓
                                  TMA
```

实际 TIRx 的 dispatcher 就是一个：

```text
(op, target)
       ↓
候选实现
       ↓
按 priority 排序
       ↓
检查 predicate
       ↓
找到第一个满足条件的实现
       ↓
生成对应 PrimFunc / IR
```

官方当前实现也是这种 target-specific、priority-ordered、predicate-guarded 的 dispatch 机制。([tvm.apache.org](https://tvm.apache.org/docs/tirx/arch/tile_dispatch.html?utm_source=chatgpt.com))

这个思想非常重要。

因为将来 NVIDIA 出：

```text
新 Tensor Core
新 TMA
新 memory
新 PTX instruction
```

不用把整个编译器重写。

只需要增加：

```text
新的 backend implementation
```

例如：

```python
@register_dispatch(
    "copy",
    "cuda",
    variant="my_new_copy",
    priority=20,
)
def my_new_copy(...):
    ...
```

这就是 TIRx 能快速适配新硬件的重要原因。

---

# 6. 真正的 TIRx lowering 是怎么进行的

官方目前的 `LowerTIRx` 非常值得以后重点读。

核心实际上可以先粗略理解成：

```text
LowerTIRx
   │
   ├── TilePrimitiveDispatch
   │
   │       copy()
   │       gemm()
   │       reduction()
   │           ↓
   │       具体硬件实现
   │
   └── LowerTIRxCleanup
           │
           └── LayoutApplier
                    ↓
             把 layout 变成具体地址计算
```

官方代码里 `LowerTIRx` 本身就是围绕这两部分展开。([tvm.apache.org](https://tvm.apache.org/docs/tirx/arch/lowering_pipeline.html))

比如：

```python
Tx.tile.copy(B, A)
```

dispatch 后可能变成类似：

```text
for i:
    B[...] = A[...]
```

或者：

```text
PTX ldmatrix
```

或者：

```text
TMA instruction
```

接着 LayoutApplier 把：

```python
B[i, j]
```

变成类似：

```text
*(B_ptr + physical_offset(i, j, lane_id, ...))
```

到了这里，高层 abstraction 基本没了。

剩下就是：

```text
loop
if
load
store
pointer arithmetic
PTX intrinsic
CUDA intrinsic
```

所以官方把 TIRx 称为 lightweight backend 是有原因的：tile primitive 被局部展开以后，剩下已经很接近原生 GPU kernel。([tvm.apache.org](https://tvm.apache.org/2026/06/22/tirx?utm_source=chatgpt.com))

---

# 7. 最后怎么变成 CUDA

后面就比较像传统编译器。

例如原来：

```text
thread_id(256)
```

lowering 后成为：

```text
threadIdx.x
```

例如：

```text
CTA id
```

成为：

```text
blockIdx.x
```

然后 TIRx PrimFunc：

```text
PrimFunc
 ├── thread bindings
 ├── loops
 ├── buffers
 └── arithmetic
```

经过 CUDA codegen：

```cpp
extern "C" __global__
void add(float* A, float* B, float* C) {
    int tx = threadIdx.x;
    C[tx] = A[tx] + B[tx];
}
```

TVM 再把 host/device 分开，GPU kernel 编译成可加载模块，host 负责 kernel launch。TVM runtime 最终把 device module 和 host module 包在一起。([github.com](https://github.com/apache/tvm/blob/main/docs/arch/codegen.rst?plain=1&utm_source=chatgpt.com))

所以可以把整个 TIRx 总结成：

```text
Python DSL
   ↓
Structured IR
   ↓
GPU-aware compiler passes
   ↓
hardware-specific dispatch
   ↓
CUDA/PTX
   ↓
GPU
```

---

# 8. 那么“从零造一个 TIRx”到底应该怎么造？

千万不要直接：

```text
fork TVM
↓
研究几万行 C++
↓
开始写 parser
↓
开始写 FFI
```

这个路线对你现在不合适。

你应该造一个：

# MiniTIRx

目标不是复刻 TVM。

目标是亲手走通：

```text
DSL
→ IR
→ Pass
→ CUDA codegen
→ GPU
```

只要这个跑通，你以后看真正 TIRx 就会完全是另一种感觉。

---

## 9. 我建议你的 MiniTIRx 最终长这样

```text
mini-tirx/
│
├── minitirx/
│   │
│   ├── ir/
│   │   ├── expr.py
│   │   ├── stmt.py
│   │   ├── buffer.py
│   │   └── function.py
│   │
│   ├── script/
│   │   └── builder.py
│   │
│   ├── passes/
│   │   ├── lower_thread.py
│   │   ├── lower_tile.py
│   │   ├── flatten_buffer.py
│   │   └── simplify.py
│   │
│   ├── backend/
│   │   └── cuda/
│   │       ├── dispatch.py
│   │       └── codegen.py
│   │
│   └── runtime/
│       └── cuda.py
│
├── examples/
│   ├── vector_add.py
│   ├── copy.py
│   └── gemm.py
│
└── tests/
```

注意这里我故意没有：

```text
C++
FFI
LLVM
复杂 AST parser
autotune
Tensor Core
TMA
```

因为第一版一个都不需要。

---

# 10. 第一步：自己定义 IR

比如先写：

```python
from dataclasses import dataclass


class Expr:
    pass


@dataclass
class Var(Expr):
    name: str


@dataclass
class Const(Expr):
    value: float


@dataclass
class Add(Expr):
    a: Expr
    b: Expr


@dataclass
class Load(Expr):
    buffer: str
    index: Expr
```

语句：

```python
class Stmt:
    pass


@dataclass
class Store(Stmt):
    buffer: str
    index: Expr
    value: Expr


@dataclass
class For(Stmt):
    var: Var
    extent: int
    body: list[Stmt]
```

函数：

```python
@dataclass
class PrimFunc:
    name: str
    args: list[str]
    body: list[Stmt]
```

现在：

```python
C[tx] = A[tx] + B[tx]
```

就能表示成：

```python
Store(
    "C",
    Var("tx"),
    Add(
        Load("A", Var("tx")),
        Load("B", Var("tx")),
    )
)
```

这一步非常重要。

因为你会第一次真正理解：

> compiler 根本不关心你原来写的代码长什么样，它只关心 IR。

---

# 11. 第二步甚至不要做 Python parser

真实 TIRx 有 TVMScript parser。

但你第一版完全不要碰。

直接做：

```python
tx = thread_id(256)

store(
    C,
    tx,
    load(A, tx) + load(B, tx)
)
```

这些函数内部只负责：

```text
创建 IR Node
```

例如：

```python
def thread_id(extent):
    return ThreadId("threadIdx.x", extent)
```

这样：

```python
with kernel("add", [A, B, C]):
    tx = thread_id(256)
    C[tx] = A[tx] + B[tx]
```

最终建立：

```text
PrimFunc
```

等整个 compiler 跑通以后，再研究：

```python
@T.prim_func
def ...
```

怎么 parse。

这是很重要的学习顺序。

---

# 12. 第三步：写你人生第一个 compiler pass

比如原始 IR：

```text
ThreadId(
    var = tx,
    extent = 256
)
```

做一个：

```python
LowerThreadBinding
```

转换为：

```text
BuiltinThreadIdxX()
```

然后 codegen 时：

```text
BuiltinThreadIdxX
```

输出：

```cpp
threadIdx.x
```

这已经是一个真正的：

```text
IR transformation pass
```

了。

你会开始理解 TVM 里面为什么到处都是：

```text
Transform
Mutator
Visitor
Pass
LowerXXX
```

---

# 13. 第四步：写 CUDA CodeGen

定义：

```python
def emit_expr(expr):
    if isinstance(expr, Var):
        return expr.name

    if isinstance(expr, Add):
        return (
            f"({emit_expr(expr.a)}"
            f" + {emit_expr(expr.b)})"
        )

    if isinstance(expr, Load):
        return (
            f"{expr.buffer}"
            f"[{emit_expr(expr.index)}]"
        )
```

Store：

```python
def emit_stmt(stmt):

    if isinstance(stmt, Store):

        return (
            f"{stmt.buffer}"
            f"[{emit_expr(stmt.index)}]"
            f" = {emit_expr(stmt.value)};"
        )
```

最后：

```python
emit_cuda(func)
```

得到：

```cpp
extern "C" __global__
void add(
    float* A,
    float* B,
    float* C
) {
    int tx = threadIdx.x;

    C[tx] = A[tx] + B[tx];
}
```

到这里，你实际上已经造出了一个：

> 极简 GPU compiler。

---

# 14. 第五步：加入 TIRx 的灵魂——TileCall

定义：

```python
@dataclass
class TileCall(Stmt):
    op: str
    src: str
    dst: str
```

用户：

```python
tile_copy(B, A)
```

先生成：

```text
TileCall
    op  = copy
    src = A
    dst = B
```

注意这里**不要立即生成 CUDA**。

而是在：

```python
LowerTile
```

里面处理。

比如：

```python
def lower_tile(op):

    if op.op == "copy":
        return make_copy_loop(
            op.src,
            op.dst
        )
```

变成：

```text
for i in range(N):
    B[i] = A[i]
```

这就已经是在模仿真正的：

```text
TilePrimitiveDispatch
```

了。

---

# 15. 然后再升级成真正的 Dispatch

下一版让 Buffer 带：

```python
Buffer(
    shape=(128, 128),
    scope="global"
)
```

以及：

```python
Buffer(
    shape=(128, 128),
    scope="shared"
)
```

dispatcher：

```text
copy(global → global)
        ↓
normal load/store

copy(global → shared)
        ↓
shared-memory copy implementation

copy(shared → register)
        ↓
register implementation
```

再进一步：

```text
target = sm_89
```

或者：

```text
target = sm_100
```

就可以做：

```python
dispatch(
    op,
    src_scope,
    dst_scope,
    target,
)
```

这已经非常接近 TIRx 的核心思想。

真实 TIRx 的 dispatch 还会结合 execution scope、layout、target、predicate 和 priority 决定具体实现。([tvm.apache.org](https://tvm.apache.org/docs/tirx/arch/tile_dispatch.html?utm_source=chatgpt.com))

---

# 16. 再往后才做 Layout

你的第一版 layout 甚至可以简单到：

```python
class Layout:

    def __init__(self, strides):
        self.strides = strides

    def index(self, i, j):

        return (
            i * self.strides[0]
            + j * self.strides[1]
        )
```

比如：

```text
shape = (4, 8)

layout:
stride = (8, 1)
```

那么：

```text
(i, j)
  ↓
i * 8 + j
```

这就是最基础 layout。

然后逐渐增加：

```text
thread axis
warp axis
lane axis
register axis
```

最后你自然就能理解 TIRx 为什么需要那么复杂的 `TileLayout`。官方 layout 本质也是把逻辑 tensor coordinate 映射到带硬件语义的物理 axis。([tvm.apache.org](https://tvm.apache.org/docs/tirx/layout.html?utm_source=chatgpt.com))

---

# 17. 你的正确开发顺序

如果是我按照你现在的基础给你设计，我会严格按照下面这个顺序，不建议跳级：

1. **手写 CUDA Vector Add**，搞懂 `threadIdx.x`、block、global memory；然后实现 MiniTIRx IR，让 `C[i]=A[i]+B[i]` 能表示成 AST/IR。接着写 builder，让 Python API 创建这棵 IR；再写 `LowerThread` 和简单的 expression simplifier。

2. **写 CUDA codegen**，把 IR 打印成 `.cu`，先做到生成的 CUDA 和你手写 CUDA 基本一样；然后接 NVRTC/CUDA Driver 或简单的编译运行层，让生成 kernel 真正在 GPU 上执行。

3. **加入 Buffer + memory scope**，支持 `global/shared/local`；之后加入 `TileCall(copy)` 和 dispatch registry，让同一个 `copy` 根据 memory scope 选择不同 lowering。

4. **加入简单 Layout**，先只支持 shape/stride，然后加入“每个线程负责哪些 element”；这时你再实现 tiled vector add、transpose、naive GEMM。

5. **实现 Tile GEMM**，先把 `tile.gemm()` lowering 成普通三层循环；之后学习 warp、shared-memory tiling，再尝试 Tensor Core intrinsic。最后才加入 predicate、priority、target-aware dispatch。

6. 当 MiniTIRx 已经能走完整链路后，再回去阅读真正 TIRx 的 `compilation_pipeline.py`、`LowerTIRx`、tile dispatch、CUDA backend。这时 PR 里面的 “add a dispatch variant”“add a lowering pass”“add PTX intrinsic” 就不会再像天书。([tvm.apache.org](https://tvm.apache.org/docs/tirx/arch/lowering_pipeline.html))

---

# 18. 最后再来看真实 TIRx 源码，你会发现它只是“大号 MiniTIRx”

可以这样一一对应：

```text
你的 MiniTIRx                 Apache TVM / TIRx

ir/                     →     tvm.tirx IR
builder/                →     TVMScript / IRBuilder
passes/                 →     tirx.transform
lower_tile.py           →     TilePrimitiveDispatch
layout.py               →     TileLayout / LayoutApplier
dispatch.py             →     tile primitive registry
cuda/codegen.py         →     CUDA source codegen
runtime/cuda.py         →     TVM runtime / CUDA runtime
```

真实 TVM 源码地图也基本按照“IR、parser/builder、transform、backend、codegen/runtime”进行组织。([tvm.apache.org](https://tvm.apache.org/docs/arch/?utm_source=chatgpt.com))

---

# 19. 为什么我不建议你现在直接研究 Blackwell GEMM

目前 `tirx-kernels` 里的高性能 kernel 主要面向 `sm_100a` Blackwell，包括 FP8/FP4 GEMM、FlashAttention 等；TIRx 编译器本身则能面向更广的 backend。([github.com](https://github.com/mlc-ai/tirx-kernels?utm_source=chatgpt.com))

你现在这台 **RTX 4080 SUPER** 更适合：

```text
Vector Add
↓
Elementwise
↓
Reduction
↓
Shared Memory
↓
Transpose
↓
Naive GEMM
↓
Tiled GEMM
↓
warp-level programming
↓
再研究 TIRx dispatch
```

而不是一开始：

```text
tcgen05
TMA
TMEM
FP4
Blackwell Tensor Core
```

否则会同时遇到：

```text
CUDA 基础
+ GPU 架构
+ compiler
+ IR
+ TVM
+ layout
+ PTX
+ Tensor Core
```

七八层概念叠在一起，很容易不知道自己究竟卡在哪。

---

# 20. 你最终真正要掌握的是这五层

以后读任何 TIRx PR，都先判断它属于哪一层：

```text
┌──────────────────────────────┐
│  DSL / Python API            │
│  用户怎么写代码               │
├──────────────────────────────┤
│  IR                          │
│  编译器怎么表示程序           │
├──────────────────────────────┤
│  Transform / Lowering        │
│  IR 怎么一步步变简单          │
├──────────────────────────────┤
│  Backend / Dispatch          │
│  怎么映射到具体 GPU 硬件      │
├──────────────────────────────┤
│  Codegen / Runtime           │
│  怎么生成 CUDA/PTX 并执行     │
└──────────────────────────────┘
```

**TIRx 的工作原理，说到底就是：**

```text
把 GPU kernel 写成结构化 IR
        ↓
保留线程、memory、layout 等硬件信息
        ↓
把重复的 tile 操作抽象成 primitive
        ↓
根据 GPU + scope + layout
选择最佳硬件实现
        ↓
lower 成普通 GPU IR
        ↓
生成 CUDA/PTX
```

这就是整个项目最核心的思想。

对于你来说，下一步最有价值的不是继续看一堆理论，而是**真的从空文件夹开始造 `mini-tirx`**。第一阶段只做：

```text
Python Builder
   ↓
Mini IR
   ↓
Vector Add
   ↓
CUDA CodeGen
   ↓
RTX 4080 SUPER 真机运行
```

等这条链跑通，我们再一层层加入 `Pass → Buffer → Shared Memory → Layout → Tile Dispatch → GEMM`。这样最后你再去看真正的 TIRx PR，会知道每一行代码到底在整个编译器的什么位置。
