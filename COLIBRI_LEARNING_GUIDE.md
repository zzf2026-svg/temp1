# 从源码彻底理解 Colibrì 大模型推理

这是一份随学习进度持续更新的中文课程笔记。课程沿用原始 `LEARNING_GUIDE.md` 的学习方式：
从一枚 token 的完整推理路径出发，逐步学习 Transformer、注意力、MLA、MoE、量化、专家权重
流式加载、推测解码和服务系统。

课程以 Colibrì 当前源码为准，并以 `c/colibri.c` 中的 GLM-5.2 引擎作为主案例。其他模型家族
只在需要比较设计差异时引入。

默认学习方式是静态阅读源码，不下载模型、不编译、不运行项目。每学完一课，再根据需要选择
小型测试或真实模型实验。

## 课程进度

- [ ] 第一课：一枚 token 是怎样生成的
- [ ] 第二课：从普通注意力推导到 MLA
- [ ] 第三课：FFN、SwiGLU 与 MoE
- [ ] 第四课：量化与整数矩阵乘法
- [ ] 第五课：模型容器与专家权重流式加载
- [ ] 第六课：缓存、预取与统一内存层级
- [ ] 第七课：采样、MTP 与推测解码
- [ ] 第八课：连续请求、KV 复用与服务架构

## 项目全景

Colibrì 的核心不是要求把超大模型完整装进 RAM 或 VRAM，而是利用 MoE 的稀疏性，把推理转化为：

```text
先计算 Router
  → 找出当前 token 需要的专家
  → 从 VRAM、RAM 或磁盘取得专家权重
  → 完成专家计算
  → 记录使用情况并改善后续放置
```

GLM-5.2 是课程的主模型。它拥有 744B 总参数，但每个 token 只激活其中一部分。Colibrì 将稠密
权重、KV 状态和热门专家保存在较快的存储层级，把大量暂时不用的路由专家保存在磁盘中。

一枚 token 在项目中的完整旅程是：

```text
文本
  → tokenizer
  → token IDs
  → embedding
  → 多层 Transformer
       → MLA attention
       → DSA 历史位置选择
       → dense FFN 或 MoE
       → 按需取得专家权重
  → final norm
  → lm_head
  → logits
  → sampling / speculative decoding
  → 新 token
  → 文本
```

主要源码入口：

| 文件 | 作用 |
|---|---|
| `c/colibri.c` | GLM-5.2 完整推理引擎，也是课程主线 |
| `c/tok.h` | tokenizer |
| `c/st.h` | 模型容器索引、tensor 元数据和按偏移读取 |
| `c/tensor.h` | 基础 tensor 描述 |
| `c/quant.h` | 量化格式与 CPU 量化计算 |
| `c/tier.h` | 专家缓存替换策略 |
| `c/sample.h` | sampling 与 stop token 管理 |
| `c/kv_persist.h` | KV 状态持久化 |
| `c/backend_cuda.*` | CUDA 后端与 VRAM 专家层级 |
| `c/backend_metal.*` | Apple Silicon Metal 后端 |
| `c/openai_server.py` | OpenAI/Anthropic 兼容 HTTP 网关 |
| `c/coli` | 用户 CLI、模型识别和资源规划 |

## 阅读原则

课程不要求从 `c/colibri.c` 第一行读到最后一行。这个文件包含成熟默认路径、平台适配和大量实验
开关，顺序通读很容易失去主线。每一课只围绕少数核心符号阅读。

阅读每个函数时先回答四个问题：

1. 输入和输出是什么，形状如何变化？
2. 它读取了哪些模型参数或持久状态？
3. 它在计算模型语义，还是只负责移动、缓存或调度数据？
4. 这个步骤主要消耗计算、内存带宽还是磁盘 I/O？

---

# 第一课：一枚 token 是怎样生成的

## 学习目标

建立最小推理骨架。暂时忽略 MLA、MoE、量化和 GPU，只理解语言模型如何根据输入生成下一枚 token。

## 1. Token 不是文字

Tokenizer 先把字符串编码为整数序列：

```text
文本 → [id₀, id₁, ..., idₙ₋₁]
```

一个 token 可能是单词、汉字、标点或字节片段。整数编号本身没有大小或距离意义。

阅读：`c/tok.h` 中的编码、解码入口，以及 `c/tests/test_tok*.c` 中的输入输出例子。

## 2. Embedding 将 ID 变成向量

模型保存一张 embedding 表：

\[
E\in\mathbb{R}^{V\times D}
\]

词表大小是 `V`，hidden size 是 `D`。token `t` 的初始表示是第 `t` 行：

\[
x=E[t]
\]

在 `c/colibri.c` 中阅读 `embed_row`，观察 `step` 怎样为输入序列准备 hidden states。

## 3. Transformer 层

先使用最小结构理解一层：

\[
x\leftarrow x+\operatorname{Attention}(\operatorname{Norm}(x))
\]

\[
x\leftarrow x+\operatorname{FFN/MoE}(\operatorname{Norm}(x))
\]

Attention 让不同位置交换信息，FFN 或 Expert 在每个位置内部变换信息，残差连接则保留原表示并
逐层增加修正。

阅读：`layer_forward`、`layers_forward`。

## 4. LM Head 与 logits

最后一个位置的 hidden state 经过 final norm 和 LM head，得到整个词表的 logits。logit 是未归一化
分数，不是 token ID，也不是直接的概率。

## 5. 自回归生成

```text
输入 prompt
  → step() 得到 logits
  → 选择一个 token
  → 把 token 加入历史
  → 再调用 step()
  → 直到 EOS、stop 或长度上限
```

阅读主调用链：

```text
main
  → model_init
  → run_text / run_serve
  → step
  → layers_forward
  → layer_forward
       → attention
       → moe
  → logits
  → sampling
```

## 第一课完成标准

能不看源码画出一枚 token 从文本到文本的完整路径，并解释 token ID、embedding、hidden state 和
logits 的区别。

---

# 第二课：从普通注意力推导到 MLA

## 学习目标

先掌握普通 Attention，再理解 GLM-5.2 为什么使用 MLA，以及 Colibrì 为什么为 prefill 和 decode
保留不同计算路径。

## 1. Q、K、V

每个位置的 hidden state 经过线性投影得到 Query、Key 和 Value：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

Query 表示当前位置想找什么，Key 表示每个位置可用什么特征被找到，Value 是找到后取回的信息。

## 2. Scaled dot-product attention

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}+M\right)V
\]

`M` 是因果 mask，禁止当前位置读取未来 token。多头注意力让模型同时学习多种相关关系。

## 3. KV Cache

Decode 每次只增加少量 token。历史位置的 K/V 不变，所以可以缓存，避免每一步重算完整上下文。
普通多头注意力需要为每个历史 token 保存大量 K/V，长上下文时占用很大。

## 4. MLA 的核心思想

MLA 不直接保存每个 head 的完整 K/V，而是保存一个共享的低维 latent。需要时再从 latent 恢复相关
表示，或者利用矩阵乘法结合律把投影吸收到 query 和输出一侧。

学习顺序：

1. 普通 MHA 的 K/V 形状；
2. KV latent 的形状；
3. RoPE 分量为什么需要单独保留；
4. prefill 为什么适合批量显式展开；
5. decode 为什么适合 weight absorption。

## 5. Weight absorption

Key 一侧的核心关系是：

\[
q^T(W_Kc)=(W_K^Tq)^Tc
\]

Value 一侧的核心关系是：

\[
\sum_t a_t(W_Vc_t)=W_V\left(\sum_t a_tc_t\right)
\]

它们让 decode 可以直接扫描低维 latent，而不必为每个历史位置展开完整 K/V。

## 6. DSA

GLM-5.2 还使用 DSA 从长历史中选择一部分位置参与注意力。先理解“选哪些历史位置”和“如何对已选
位置计算 Attention”是两个不同步骤，再阅读 `c/colibri.c` 中的 DSA 选择路径。

## 源码入口

- `attention_rows`、`attention`；
- absorb 相关分支；
- `c/tests/test_dsa_select.c`；
- `c/tests/test_absorb_determinism.cu`。

## 第二课完成标准

能解释普通 KV Cache 为什么大、MLA 保存了什么、RoPE 分量为什么特殊，以及 absorption 怎样减少
decode 的中间数据和内存流量。

---

# 第三课：FFN、SwiGLU 与 MoE

## 学习目标

理解 Attention 之后为什么还需要 FFN，以及 MoE 如何把一个 FFN 扩展为许多可选择的专家。

## 1. FFN 与非线性

一个简单 FFN 是：

\[
\operatorname{FFN}(x)=W_2\,\sigma(W_1x)
\]

Attention 主要负责跨位置取回信息，FFN 负责逐位置进行非线性特征变换。

## 2. SwiGLU

现代模型常使用 gate、up、down 三个投影：

\[
y=W_{down}\left(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right)
\]

学习 gate 怎样控制信息通过，up 怎样扩展中间维度，down 怎样投影回 hidden size。

## 3. Expert

一个 expert 本质上就是一套独立的 SwiGLU FFN 权重。MoE 层拥有很多 experts，但一个 token 通常
只运行 Router 选中的少数几个。

## 4. Router 与 top-K

```text
hidden state
  → Router 得到所有专家分数
  → 选择 top-K
  → 执行这些 experts
  → 按路由权重合并输出
```

同时学习 shared expert：它不参与普通 top-K 竞争，而是始终执行，为所有 token 提供共享能力。

## 5. 总参数与激活参数

744B 描述完整模型保存的总参数量，不代表每个 token 都计算 744B 参数。MoE 降低每步计算量，却没有
消除所有专家的存储需求。

## 6. Batch expert union

一个 batch 中多个 token 可能选择同一专家。引擎先求专家并集，可以让同一批中的位置共享一次加载，
避免重复读取权重。

## 源码入口

- `moe`；
- Router top-K 与输出合并位置；
- dense/shared expert 分支；
- `expert_load` 留到第五课详细学习。

## 第三课完成标准

能手算一个 4 选 2 的 Router 例子，解释 expert 输出怎样合并，以及 MoE 为什么从“纯计算问题”变成
“计算 + 权重移动问题”。

---

# 第四课：量化与整数矩阵乘法

## 学习目标

理解 Colibrì 怎样减少权重体积和搬运字节，以及低位权重如何真正参与矩阵乘法。

## 1. 数值格式

依次理解 FP32、BF16/FP16、FP8、INT8 和 INT4。重点不是背诵格式位域，而是理解表示范围、精度、
存储大小与计算支持之间的权衡。

## 2. Scale 与量化误差

最小量化模型：

\[
q=\operatorname{round}(x/s),\qquad \hat{x}=q\cdot s
\]

学习 per-tensor、per-row 和 group scale 的差异。scale 粒度越细通常越准确，但会增加元数据和实现
复杂度。

## 3. INT4 打包

一个 int4 只占半个字节，两个值可以打包进一个 byte。学习 signed/unsigned 解释、零点、符号扩展和
字节顺序，理解为什么读取正确字节还不等于解释正确格式。

## 4. 量化矩阵乘法

理解两条常见路线：

```text
低位权重 → 临时反量化 → 浮点乘加

低位权重 + 量化 activation
  → 整数点积
  → 应用 scale
  → 浮点输出
```

随后理解 SIMD、AVX2/AVX-512、NEON、IDOT 和 GPU kernel 只是在保持同一数学语义的前提下，提高
并行度和数据吞吐。

## 源码入口

- `docs/FORMATS.md`；
- `c/quant.h`；
- `c/native_quant*.h`；
- `c/tests/test_int_kernel_exact.c`；
- `c/tests/test_i4_grouped.c`、`test_idot.c`、`test_fp8_load.c`。

## 第四课完成标准

能手算一组小数的量化与恢复，解释 scale 的作用，并说明“模型缩小”“矩阵乘法变快”和“生成质量
保持不变”为什么是三个需要分别验证的结论。

---

# 第五课：模型容器与专家权重流式加载

## 学习目标

理解数百 GB 模型怎样组织在文件中，以及引擎如何只读取当前需要的 tensor 或 expert。

## 1. Tensor 元数据

模型容器至少需要描述：

```text
tensor 名称
shape
数据类型或量化格式
scale 规则
文件偏移
数据长度
```

引擎根据元数据定位权重，而不是从文件开头顺序扫描整个模型。

## 2. 稠密权重与路由专家

Embedding、Attention、Router、shared expert 等稠密部分会被频繁使用，适合常驻较快层级。大量 routed
experts 只有被 Router 选中时才需要，可以留在磁盘并按需加载。

## 3. 一次专家加载

```text
Router 选中 expert
  → 查询 resident/cache 状态
  → 命中：直接取得权重
  → 未命中：根据容器偏移读取
  → 校验并解释量化格式
  → 放入可执行 slot
  → 完成 gate/up/down 计算
```

## 4. 读取方式

学习 `pread` 为什么适合按确定偏移读取，mmap 如何利用操作系统 page cache，以及 O_DIRECT 为什么可以
绕过一部分缓存行为。此时只理解语义差异，不急着比较谁一定更快。

## 源码入口

- `c/tensor.h`；
- `c/st.h`；
- `model_init`、`expert_load_impl`、`expert_load`；
- `c/tests/test_st*.c`；
- `c/tests/test_expert_store_ops.c`。

## 第五课完成标准

能从“tensor 名称”一路解释到“文件中的一段字节怎样成为一次 expert matmul 的权重”，并说明为什么
完整模型不必同时驻留内存。

---

# 第六课：缓存、预取与统一内存层级

## 学习目标

理解 Colibrì 最具代表性的系统设计：把 VRAM、RAM 与磁盘视为速度和容量不同的权重层级。

## 1. Cache 与 LRU

专家缓存保存最近或经常使用的权重。命中可以避免磁盘读取；空间不足时，LRU 优先淘汰最久没有使用
的内容。

先手算一个容量为 3、访问序列为 `A B C A D B` 的 LRU 例子，再阅读 `c/tier.h`。

## 2. 热门专家固定

使用历史可以帮助引擎识别高频专家并将其固定在 RAM 或 VRAM。它对重复负载可能有效，但历史也可能
对某类 prompt 过拟合，因此必须在留出负载上验证。

## 3. 预取

预取不是减少权重大小，而是把读取提前并尝试与当前计算重叠：

```text
计算当前层
  ║
  ╚═ 同时预测并读取下一层专家
```

预测正确时减少等待；预测错误时会浪费带宽，甚至挤出真正需要的缓存内容。

## 4. 异步 I/O 与流水线

学习 worker pipe、io_uring、读取队列和 batch union 如何让常驻 expert 的计算与缺失 expert 的加载
并行。区分“总工作减少”和“工作量不变但关键路径缩短”。

## 5. VRAM、RAM 与磁盘

同一 expert 无论从哪个层级取得，都应包含相同权重。正常放置策略只决定速度，不应改变 Router 决策或
模型精度。涉及减少 top-K 或修改路由的实验，必须与这种语义保持型优化分开讨论。

## 源码入口

- `c/tier.h`；
- `expert_is_resident`、`expert_prefetch`、`pipe_dispatch`；
- pilot 与 io_uring 相关路径；
- `docs/tuning.md`、`docs/benchmarks.md`；
- `c/tests/test_tier.c`、`test_pilot_ring.c`、`test_uring.c`。

## 第六课完成标准

能画出一次缓存命中和一次磁盘未命中的时间线，并解释命中率、磁盘带宽、预取准确率和 token/s 之间
为什么不是简单的一一对应关系。

---

# 第七课：采样、MTP 与推测解码

## 学习目标

理解模型怎样从 logits 选择 token，以及为什么一次 forward 有时可以尝试生成多个 token。

## 1. Softmax 与 temperature

\[
p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
\]

temperature 较低时分布更集中，较高时分布更平缓。greedy 总是选择最高分，随机采样则根据概率分布
选择。

## 2. Top-K 与 top-p

Top-K 只保留分数最高的 K 个 token；top-p 保留累计概率达到阈值的最小候选集合。学习它们如何限制
长尾候选，同时注意 token sampling 与 MoE expert top-K 是两件不同的事。

## 3. MTP

GLM-5.2 的 MTP head 可以根据当前状态提出后续 token 草稿。主模型随后批量计算这些位置并验证草稿。

## 4. 推测解码状态机

```text
主模型得到当前状态
  → draft 提出若干 token
  → 主模型批量验证
  → 接受连续正确前缀
  → 在第一个不一致位置使用主模型结果
  → 修正 KV、采样和请求状态
```

必须理解：草稿和验证如果因为 batch size、kernel、缓存状态或路由策略计算了不同函数，即使程序没有
崩溃，接受率和最终语义也会失去可信度。

## 5. 是否真的更快

推测解码的收益取决于接受率、验证成本、专家命中率和批次专家并集。接受率高并不自动等于端到端
token/s 更高。

## 源码入口

- `c/sample.h`；
- `step` 的多位置 logits 路径；
- `spec_decode`；
- `c/decode_batch.h`；
- `c/tests/test_spec_decode_state.c`、`test_topp.c`、`test_sample_nan.c`。

## 第七课完成标准

能手工推演一次“草稿 3 个、接受前 2 个”的验证过程，并说明 sampling 随机性、KV 状态和专家路由
为何都必须保持一致。

---

# 第八课：连续请求、KV 复用与服务架构

## 学习目标

把单次 token 生成扩展为可持续运行的聊天和 API 服务。

## 1. Chat 与 Serve

`coli chat` 维护一个交互式会话；`coli serve` 启动 C 引擎和 Python HTTP gateway，对外提供 OpenAI/
Anthropic 兼容协议；`coli web` 在相同服务基础上增加浏览器界面。

## 2. KV 复用

新请求如果与已有上下文共享前缀，可以复用相应 KV 状态，避免重新 prefill。学习以下边界：

- 最长公共前缀在哪里结束；
- 新 token 从哪个位置开始计算；
- tokenizer 或模板变化为什么会使前缀失效；
- 持久化 KV 怎样在重启后恢复暖会话。

## 3. KV Slots

服务同时维护多个独立上下文时，需要为不同请求分配 KV slot。每个 slot 都有自己的 token 历史、KV
状态、采样状态和停止条件，不能相互污染。

## 4. 连续请求与批处理

不同请求可能拥有不同历史长度并处于不同生成阶段。学习 ragged batch 如何表示这些差异，以及 batch
怎样影响 Attention、expert union 和吞吐。

## 5. 服务协议

```text
HTTP JSON 请求
  → openai_server.py 解析与 prompt 构造
  → C 引擎 serve 协议
  → tokenizer / prefill / decode
  → 流式 token 与 telemetry
  → OpenAI/Anthropic 响应事件
  → 客户端或 Web UI
```

同时学习取消、客户端断开、stop 条件、子进程监督和资源释放。这些不改变模型数学，却决定服务能否
长期正确运行。

## 源码入口

- `run_serve`、`run_serve_mux`；
- `c/serve_codec.h`；
- `c/kv_persist.h`、`c/kv_prefix.h`；
- `c/openai_server.py`；
- `docs/api.md`；
- `c/tests/test_openai_server.py`、`test_serve_codec.c`、`test_stop_*.py`。

## 第八课完成标准

能画出一个聊天请求从 HTTP JSON 到流式 token 的完整时序图，并解释模型计算、KV 会话状态、协议转换
和 Web 展示分别由哪一层负责。

---

# 课程结束后的项目全景

完成八课后，再回头阅读 Colibrì，可以把系统拆成四层：

```text
模型语义层
  Tokenizer、Attention、MLA、MoE、Sampling

数值计算层
  Tensor、量化格式、CPU SIMD、CUDA、Metal、Vulkan

存储与调度层
  模型容器、Expert Store、LRU、Pin、Prefetch、异步 I/O

产品与服务层
  coli CLI、资源规划、OpenAI gateway、KV slots、Web telemetry
```

这四层相互影响，但验证标准不同：

- 模型语义层首先要求结果正确；
- 数值计算层要求与 reference 或 oracle 一致；
- 存储与调度层要求保持语义并缩短端到端关键路径；
- 服务层要求请求隔离、协议正确和生命周期可靠。

## 推荐学习节奏

| 周次 | 课程 | 主要产出 |
|---|---|---|
| 第 1 周 | 第一课 | 一枚 token 的完整流程图 |
| 第 2 周 | 第二课 | MHA、MLA 与 KV Cache 对比图 |
| 第 3 周 | 第三课 | SwiGLU 和 4 选 2 Router 手算 |
| 第 4 周 | 第四课 | INT4 量化与矩阵乘法笔记 |
| 第 5 周 | 第五课 | 从容器偏移到 expert matmul 的数据流 |
| 第 6 周 | 第六课 | LRU 与预取流水线推演 |
| 第 7 周 | 第七课 | 推测解码状态机 |
| 第 8 周 | 第八课 | API 请求和 KV slot 时序图 |

## 每课笔记模板

```text
本课解决的问题：
核心概念：
关键公式及每个符号的含义：
输入、输出与形状变化：
对应源码符号：
主要计算或数据移动：
必须保持的不变量：
我能手算的最小例子：
仍然不理解的问题：
```

学完这条主线后，再根据兴趣进入 GPU kernel、DeepSeek V4、Kimi K3、多 SSD、集群专家计算或 Web
控制台等专题。它们是对八课知识的扩展，不需要提前塞进基础课程。
