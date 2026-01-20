> [English](README.md)

# 多头注意力（MHA）变体

该变体增加了一个**朴素、显式**的多头因果自注意力（multi-head causal self-attention）实现。

代码：
- `src/variants/mha/mha_attention.h`
- `src/variants/mha/mha_attention.cpp`

## 相比基线改变了什么？

基线实现（`nn::self_attention_1h`）把通道维度当作一个宽度为 $C$ 的单头。

该变体（`nn::variants::mha::self_attention_mha`）保持**相同的参数布局**：
- `w_qkv: [C,3C]`, `b_qkv: [3C]`
- `w_proj: [C,C]`, `b_proj: [C]`

……但改变了对 $C$ 的**解释方式**：

- 选择 `n_heads = H`
- 要求 $C = H \cdot D$，其中 $D$ 是每个 head 的维度
- 先计算出 `q,k,v`（形状 `[B,T,C]`），然后把每个都 reshape 为 `[B,T,H,D]`

接着对每个 head $h$ 独立计算注意力：

$$
S_{h}[i,j] = \frac{Q_{h}[i] \cdot K_{h}[j]}{\sqrt{D}} + \text{causalMask}(i,j)
$$

$$
P_{h}[i,:] = \mathrm{softmax}(S_{h}[i,:])
$$

$$
Y_{h}[i] = \sum_{j} P_{h}[i,j] V_{h}[j]
$$

把各个 head 拼回 `[B,T,C]`（实现上是把 `[B,T,H,D]` reshape 回 `[B,T,C]`），再做输出投影。

## 为什么要有这个变体

- 它是一个**教学优先的模板**：展示如何添加变体
	- 在 `src/variants/...` 下新增一个模块
	- 在 `docs/variants/...` 下新增一小组配套文档
	- 加一个聚焦的测试来证明关键正确性
- 它把索引写得很显式，便于后续：
	- 验证正确性
	- 并排加入 KV-cache / GQA / RoPE
	- 在不改变语义的前提下，将 kernel 替换到后端边界之后
