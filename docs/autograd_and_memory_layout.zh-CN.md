> [English](autograd_and_memory_layout.md)

# Autograd + 内存布局（如何像读论文一样读 C++）

这篇文档覆盖两件在论文里常常是“默认你知道”、但在从零 C/C++ 实现里极其关键的内容：

1) **张量如何在内存中存储**（flat array + indexing）
2) **本仓库的 reverse-mode autograd 如何工作**（计算图构建 + backward 执行）

配套阅读：
- `docs/transformer_math_to_code.md`（公式 → 函数/文件）
- `docs/training_and_inference_walkthrough.md`（运行时流程）


## Part A — 张量存储与索引

### A.1 存储

`nn::Tensor` 包含：
- `data`：`shared_ptr<vector<float>>`（连续的 flat buffer）
- `shape`：`vector<int>`（例如 `{B,T,C}`）

buffer 使用 **row-major（行优先）**布局（C/C++ 默认），也就是 **最后一维连续**。

### A.2 通用 row-major 索引

对形状 `[d0,d1,...,d_{k-1}]` 与索引 `[i0,i1,...,i_{k-1}]`，其 flat offset：

$$\text{offset} = (((i_0\cdot d_1 + i_1)\cdot d_2 + i_2)\cdots)\cdot d_{k-1} + i_{k-1}$$

本仓库里最常用的特例：

- `[B,T,C]`（hidden states）
  - `offset = (b*T + t)*C + c`
- `[B,T,T]`（attention scores/probs）
  - `offset = (b*T + i)*T + j`
- `[M,N]`（matmul）
  - `offset = i*N + j`

### A.3 例子：`[B=2,T=3,C=4]`

令 `X` 形状为 `[2,3,4]`。

- `X[0,0,:]` 的起始 offset：`(0*3+0)*4 = 0`
- `X[0,1,:]` 的起始 offset：`(0*3+1)*4 = 4`
- `X[1,0,:]` 的起始 offset：`(1*3+0)*4 = 12`

因此每个 time step 都是一段长度为 `C` 的连续 float。

这也是为什么很多算子会外层循环 `b`、`t`，而内层紧密循环 `c`（连续内存）。

### A.4 本仓库里的 reshape 语义

`nn::reshape(x, new_shape)`：
- **共享**同一个 `data` buffer（view）
- 创建一个新的 `Tensor` 对象并设置新的 `shape`
- 如果启用了梯度，它会把梯度 1:1 路由回父张量（对 flat grad buffer 做对应累加）

学习上的重要点：
- 本仓库没有 stride 元数据；reshape 只有在元素总数完全一致时才合法。


## Part B — Autograd 引擎：调用 backward 会发生什么？

### B.1 `Tensor` 会携带哪些信息

一个 tensor 可能包含：
- `requires_grad`（bool）
- 当 `requires_grad=true` 时的 `grad` buffer（大小与 `data` 相同）
- 可选的 `node` 指针，用来描述“它是怎么被算出来的”

`node` 包含：
- `parents`：父张量列表
- `backward`：一个函数/lambda，它消费 `out.grad` 并把梯度累加到每个 parent 上

### B.2 算子什么时候会构建计算图？

大多数算子都会调用类似 `want_grad(t)` 的内部判断：
- 全局 grad mode 开启（`nn::is_grad_enabled()` 为 true）
- 并且该 tensor `requires_grad=true`

因此，是否构建计算图由两层开关控制：
- **全局模式**：`nn::GradMode no_grad(false);` 可以在一个作用域内关闭图构建
- **tensor 级别**：`requires_grad`

在生成（inference）时，代码会使用 `nn::GradMode no_grad(false);`，因此推理不会构建计算图。

### B.3 backward 的执行模型

调用 `loss.backward()` 大致做：

1) 确保 `loss` 是标量；把 `loss.grad = 1` 作为种子
2) 构建一个 **拓扑序**（parents 在 children 之前）
3) 按拓扑序的逆序遍历，执行每个 node 的 `backward` lambda

实现位置：
- `nn::Tensor::backward()`（`src/tensor.cpp`）

关键点：
- 对 parent 的梯度是 **累加**（`+=`）到 grad buffer 里的。
- 这与论文里“多条路径的梯度求和”是一回事：在代码里你会看到实打实的 `+=`。

### B.4 例子：`add`

前向：
$$y = a + b$$

反向：
$$\frac{\partial L}{\partial a} += \frac{\partial L}{\partial y}$$
$$\frac{\partial L}{\partial b} += \frac{\partial L}{\partial y}$$

代码里：
- `nn::add(...)`（`src/ops.cpp`）会创建一个 node，parents 为 `{a,b}`，其 backward 会执行：
  - `(*a.grad)[i] += (*out.grad)[i]`
  - `(*b.grad)[i] += (*out.grad)[i]`

这是一个非常好的“论文公式 → C++ loop”入门例子。

### B.5 例子：`matmul2d`

前向：
$$O = A B$$

反向（矩阵微积分）：
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} B^T$$
$$\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial O}$$

代码里：
- `nn::matmul2d(...)`（`src/ops.cpp`）用显式 for-loop 计算上面两条公式。

学习建议：
- 对照 loop 的下标与 shape：
  - `dA[i,kk]` 对 `j` 求和
  - `dB[kk,j]` 对 `i` 求和

### B.6 例子：softmax 反向（Jacobian-vector product）

softmax：
$$y = \mathrm{softmax}(x)$$

完整 Jacobian 很贵，所以使用常见高效恒等式：
$$\frac{\partial L}{\partial x} = y \odot \left(\frac{\partial L}{\partial y} - \langle \frac{\partial L}{\partial y}, y \rangle\right)$$

代码里：
- `nn::softmax_lastdim(...)`（`src/ops.cpp`）

### B.7 例子：交叉熵反向

对 logits $z$、softmax 概率 $p$ 与目标类别 $y$：
$$\frac{\partial L}{\partial z_v} = \frac{1}{N}(p_v - \mathbb{1}[v=y])$$

代码里：
- `nn::cross_entropy(...)`（`src/ops.cpp`）会在 forward 里保存 probs，并在 backward 中用上式计算梯度。


## Part C — 注意力的内存布局与 loops（你应该重点观察什么）

单头注意力会构造：
- `scores`：`[B,T,T]`
- `probs`：`[B,T,T]`
- `v`：`[B,T,C]`
- 输出 `att`：`[B,T,C]`

关键 loops 直接对应数学：

1) scores：
$$S[b,i,j] = (Q[b,i,:]\cdot K[b,j,:]) / \sqrt{C}$$

2) 对 `j`（keys 维度）做 softmax

3) 加权求和：
$$Y[b,i,c] = \sum_j P[b,i,j] V[b,j,c]$$

代码位置：
- `nn::self_attention_1h(...)`（`src/ops.cpp`）

学习建议：
- 时刻追踪“哪个维度在内存里是连续的”：
  - `[B,T,C]` 中 `c` 连续
  - `[B,T,T]` 中固定 `(b,i)` 后，`j` 连续

这在未来做性能优化（SIMD / blocking / GPU kernels）时会变得非常关键。


## Part D — 未来特性的“文档契约”

这是一个学习型项目。每当新增一个特性，我们也应同步更新/补充文档，让读者能从：

- **论文概念/公式** → **精确函数** → **精确张量形状** → **实现选择与权衡**

形成闭环。

每个新特性的实用 checklist：

1) 更新 `docs/transformer_math_to_code.md`
- 添加公式并列出新增/修改的函数。

2) 如果特性较大，添加独立文档
- 例如：
  - `docs/multihead_attention.md`
  - `docs/kv_cache.md`
  - `docs/rlhf_overview.md`（如果未来会做的话）

3) 添加有针对性的代码注释
- 更偏好在核心函数顶部写短的“数学块”注释。
- 更偏好像 `[B,T,C]` 这样的 shape 标注。

4) 如果运行时流程有变化，更新 `docs/training_and_inference_walkthrough.md`
- 例如：新的推理路径、cache、新的 checkpoint 字段。

5) 当数学/语义变化时，增加/扩展测试
- 至少要有一个回归测试，防止“悄悄破坏正确性”。


## 参考（论文名）

- “Attention Is All You Need”（scaled dot-product attention）
- “Language Models are Unsupervised Multitask Learners”（GPT-2 风格 decoder-only LM 目标）

（这里只引用论文名用于术语统一与定位；本文档是对本仓库实现的原创讲解。）
