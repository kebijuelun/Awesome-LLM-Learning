# Awesome-LLM-Learning [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
## 引言
欢迎来到本 Repo！这里是一个专注于大语言模型学习的仓库，旨在为大语言模型学习入门者和大语言模型研发岗位的面试准备者提供全面的基础知识。

我们涵盖了多个主题，包括深度学习基础知识、自然语言处理基础知识、大语言模型基础知识以及大语言模型推理和应用等内容。你将在这里找到丰富的学习资料，帮助你构建对大语言模型的全面理解和应用能力。

除了基础知识，我们还会推荐一些大语言模型领域的前沿论文，帮助你跟上最新的研究进展，掌握最先进的技术。

无论你是新手还是有经验的从业者，本 Repo 都会为你提供有价值的学习资源。欢迎一起探索大语言模型的奥秘，让我们一同进步和成长！如果你有任何问题或建议，欢迎提出，我们将尽力完善这个仓库，为大家提供更好的学习体验。谢谢你的参与！
## 目录
- [1.深度学习基础知识](1.深度学习基础知识)
	- [1.Transformer基础](1.深度学习基础知识/1.Transformer基础)
	- [2.深度神经网络基础](1.深度学习基础知识/2.深度神经网络基础)
- [2.自然语言处理基础知识](2.自然语言处理基础知识)
	- [1.分词器(Tokenizer)](2.自然语言处理基础知识/1.分词器(Tokenizer))
	- [2.经典NLP模型](2.自然语言处理基础知识/2.经典NLP模型)
	- [3.困惑度(perplexity)](2.自然语言处理基础知识/3.困惑度(perplexity))
- [3.大语言模型基础知识](3.大语言模型基础知识)
	- [1.训练框架介绍(Megatron-lm、DeepSpeed)](3.大语言模型基础知识/1.训练框架介绍(Megatron-lm、DeepSpeed))
	- [2.参数高效微调(PEFT)](3.大语言模型基础知识/2.参数高效微调(PEFT))
	- [3.经典开源LLM介绍](3.大语言模型基础知识/3.经典开源LLM介绍)
	- [4.RLHF介绍](3.大语言模型基础知识/4.RLHF介绍)
	- [5.CoT、ToT、GoT介绍](3.大语言模型基础知识/5.CoT、ToT、GoT介绍)
	- [6.SFT训练](3.大语言模型基础知识/6.SFT训练)
- [4.大语言模型推理](4.大语言模型推理)
	- [1.Huggingface推理参数介绍](4.大语言模型推理/1.Huggingface推理参数介绍)
- [5.大语言模型应用](5.大语言模型应用)
	- [1.LangChain介绍](5.大语言模型应用/1.LangChain介绍)
- [6.大语言模型前沿分享](6.大语言模型前沿分享)
	- [1.LLM相关博客分享](6.大语言模型前沿分享/1.LLM相关博客分享)
	- [2.LLM相关论文分享](6.大语言模型前沿分享/2.LLM相关论文分享)
## 深度学习基础知识
### Transformer基础
#### Transformer 的 Self Attention Layer 实现

- Transformer 模型的自注意力（Self Attention）机制涉及三种线性变换：查询（Q）、键（K）、值（V），以及通过计算注意力权重和加权求和来生成最终的输出。以下是自注意力机制的公式：

  - 假设输入序列为 $X \in \mathbb{R}^{L \times d}$，其中 $L$ 是序列长度，$d$ 是特征维度。自注意力层中的线性变换矩阵为 $W_q \in \mathbb{R}^{d \times d_k}$、$W_k \in \mathbb{R}^{d \times d_k}$ 和 $W_v \in \mathbb{R}^{d \times d_v}$，其中 $d_k$ 和 $d_v$ 是分割查询、键和值时的维度。
  - 查询（Q）、键（K）和值（V）的线性变换：

    - $Q=X⋅W_{q}$
    - $K=X⋅W_{k}$
    - $V=X⋅W_{v}$
  - 多头拆分：

    - 在多头自注意力中，将查询（Q）、键（K）和值（V）分成 $h$ 个头，其中 $h$ 是注意力头的数量。对于每个头，使用不同的线性变换权重。
  - 注意力得分计算：

    - 使用查询（Q）和键（K）计算注意力得分，可以使用点积注意力（Scaled Dot-Product Attention）：
      - $Attention(Q,K)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})$
  - 注意力加权求和：

    - 使用注意力权重对值（V）进行加权求和，得到自注意力的输出：
      - $Output=Attention(Q,K)V$
  - 上述公式涵盖了Transformer模型中自注意力层的基本计算步骤。在实际应用中，通常还会考虑添加缩放（scaling）、掩码（masking）等操作来优化注意力机制的计算过程。多头自注意力则在每个头上分别进行这些计算，然后将多个头的输出拼接或串联起来，进一步提取序列中的信息
- 一个多头 Self Attention 代码实现如下：

```python
import torch
import torch.nn.functional as F

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换矩阵
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attention加权求和
        output = torch.matmul(attn_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        return output

# 测试自注意力层
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 16

input_data = torch.randn(batch_size, seq_len, d_model)
self_attention = SelfAttentionLayer(d_model, num_heads)
output = self_attention(input_data)
print(output.size())  # 输出: torch.Size([16, 10, 512])
```

#### Transformer 中 Self Attention 计算为什么要除以特征维度数量的平方根

[参考1](https://zhuanlan.zhihu.com/p/366592542), [参考2](https://www.cnblogs.com/hisi-tech/p/16709612.html), [参考3](https://zhuanlan.zhihu.com/p/105722023)

- 上面 Transformer 代码中有使用 $\sqrt{d_{k}}$ 来对 softmax 之后的结果进行 scale，解释如下：

  > We suspect that for large values of $d_{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{d_{k}}$. To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product $q ⋅ k=\sum_{i=1}^{d_{k}}(q_{i}k_{i})$, has mean 0 and variance $d_{k}$.
  >

  1. 首先要除以一个数，防止输入 softmax 的值过大，导致偏导数趋近于 0；
  2. 选择根号 d_k 是因为可以使得 q*k 的结果满足期望为 0，方差为 1 的分布，类似于归一化。

#### 为什么要引入位置编码 (Position Embedding)

[参考](https://zhuanlan.zhihu.com/p/106644634)

- 对于任何一门语言，单词在句子中的位置以及排列顺序是非常重要的，它们不仅是一个句子的语法结构的组成部分，更是表达语义的重要概念。一个单词在句子的位置或排列顺序不同，可能整个句子的意思就发生了偏差。

  > I do not like the story of the movie, but I do like the cast.
  > I do like the story of the movie, but I do not like the cast.
  > 上面两句话所使用的的单词完全一样，但是所表达的句意却截然相反。那么，引入词序信息有助于区别这两句话的意思。
  >
- Transformer 模型抛弃了 RNN、CNN 作为序列学习的基本模型。我们知道，循环神经网络本身就是一种顺序结构，天生就包含了词在序列中的位置信息。当抛弃循环神经网络结构，完全采用 Attention 取而代之，这些词序信息就会丢失，模型就没有办法知道每个词在句子中的相对和绝对的位置信息。因此，有必要把词序信号加到词向量上帮助模型学习这些信息，位置编码（Positional Encoding）就是用来解决这种问题的方法。

#### RoPE 位置编码

[参考1](https://zhuanlan.zhihu.com/p/359502624)， [参考2](https://blog.csdn.net/weixin_43646592/article/details/130924280)，[参考3](https://zhuanlan.zhihu.com/p/642289220)

- RoPE
  - 二维情况下用复数表示的 RoPE
    ![在这里插入图片描述](https://img-blog.csdnimg.cn/b3760f402fe84096b8c01be93f1d8fe5.png)
    根据复数乘法的几何意义，该变换实际上对应着向量的旋转，所以我们称之为“旋转式位置编码”
  - RoPE通过绝对位置编码的方式实现相对位置编码，综合了绝对位置编码和相对位置编码的优点
    - 绝对位置编码：最原始的正余弦位置编码（即sinusoidal位置编码）是一种绝对位置编码，但从其原理中的正余弦的和差化积公式来看，引入的其实也是相对位置编码。
      - 优势： 实现简单，可预先计算好，不用参与训练，速度快
      - 劣势： 没有外推性，即如果预训练最大长度为512的话，那么最多就只能处理长度为512的句子，再长就处理不了了。当然，也可以将超过512的位置向量随机初始化，然后继续微调
    - 相对位置编码：经典相对位置编码RPR式
      - 直接地体现了相对位置信号，效果更好。具有外推性，处理长文本能力更强
  - 主要就是对attention中的q, k向量注入了绝对位置信息，然后用更新的q,k向量做attention中的内积就会引入相对位置信息了

#### Alibi 位置编码

[参考1](https://paperswithcode.com/method/alibi)， [参考2](https://zhuanlan.zhihu.com/p/525552086)， [参考3](https://zhuanlan.zhihu.com/p/642846676)

- ALiBi (Attention with Linear Biases，22年ICLR)，是一种 position embedding 方法，允许 Transformer 语言模型在推理时处理比其训练时更长的序列。
  - ALiBi 在不使用实际位置嵌入的情况下实现这一目标。相反，ALiBi计算某个键和查询之间的注意力时，会根据键和查询的距离来对查询可以分配给键的注意力值进行惩罚。因此，当键和查询靠近时，惩罚非常低，而当它们相距较远时，惩罚非常高。这个方法的动机很简单，即靠近的单词比远离的单词更重要。
    - 不添加position embedding，然后添加一个静态的不学习的 bias
      $softmax(q_{i}K^{T} + m \cdot [-(i-1,...,-2,-1,0)])$
  - ALiBi方法的速度与正弦函数嵌入或绝对嵌入方法相当（这是最快的位置编码方法之一）。在评估超出模型训练序列长度的序列时，ALiBi优于这些方法和Rotary嵌入（这称为外推）方法（ALiBi的方式，训练快了11%，并且会减少11%的内存消耗）。
  - position embedding 并没有加在 work embedding 上，而是加在了 Q*K^T 上面

#### PI 位置插值支持长度外推

[参考](https://zhuanlan.zhihu.com/p/645263524)

- PI位置插值
  - 论文名称：EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION
  - 论文链接：https://arxiv.org/pdf/2306.1559
  - 方法：位置插值线性向下缩放了输入位置索引以匹配原始的上下文窗口大小，而不是外推超过训练时所用的上下文长度，因为这可能会导致灾难性的较高的注意力分数，从而完全破坏了自注意力机制。
    ![PI interpolation](./1.深度学习基础知识/images/PI.png)
    - 左半部分为预训练阶段的位置向量范围[0,2048]
    - 右上角为长度外推的部分(2048,4096]
    - 左下角为位置插值法，将[0,4096]的值降采样到[0,2048]预训练阶段支持的范围
  - 论文的实现很简单，只需要将对应位置缩放到原先支持的区间([0,2048])内：计算公式如下，$L$ 为原先支持的长度(如2048)，$L^{'}$ 为需要扩展的长度(如4096)：
    $f^{'}(x, m)=f(x, \frac{mL}{L^{'}})$

#### NTK-Aware Scaled RoPE 支持长度外推

[参考1](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), [参考2](https://zhuanlan.zhihu.com/p/645263524)

- 在这项工作中，作者针对当前RoPE插值方法的不足，提出了一种改进方案。通过应用神经切线核（NTK）理论，作者发现现有的线性插值方法在处理距离接近的标记时存在局限性。因此，作者设计了一种非线性插值方案，以改变RoPE的基数。这种方法在保持位置信息完整的同时，有效地提高了上下文大小。实验证明，该方法在没有进行模型微调的情况下就能显著减小困惑度，成为一种非常有效的优化策略。作者相信，通过进一步的微调，这个方法的效果将得到更好的提升。

#### Pre Norm 与 Post Norm 的区别？

参考：[为什么 Pre Norm 的效果不如 Post Norm？](https://kexue.fm/archives/9009)

- 定义：
  - Pre Norm（Norm and add）： $x_{t+1} = x_{t} + F_{t}(Norm(x_{t}))$
  - Post Norm（Add and Norm）： $x_{t+1} = Norm(x_{t} + F_{t}(x_{t})$
- 在同一训练设置下，同一设置之下，Pre Norm 结构往往更容易训练，但最终效果通常不如 Post Norm：
  - Pre Norm 更容易训练好理解，因为它的恒等路径更突出
  - Pre Norm 的深度有“水分”！也就是说，一个L层的Pre Norm模型，其实际等效层数不如 L 层的 Post Norm 模型，而层数少了导致效果变差了。Post Norm 每 Norm 一次就削弱一次恒等分支的权重，所以Post Norm反而是更突出残差分支的，因此 Post Norm 中的层数更加“足秤”，一旦训练好之后效果更优。
- Post Norm 的结构迁移性能更加好，也就是说在 Pretraining 中，Pre Norm 和 Post Norm 都能做到大致相同的结果，但是 Post Norm 的 Finetune 效果明显更好

### 深度神经网络基础
#### BN 和 LN 的区别

- BN 与 LN 定义：Batch Normalization 是对这批样本的同一维度特征做规范化处理， Layer Normalization 是对这单个样本的所有维度特征做规范化处理
![BN介绍](https://img-blog.csdnimg.cn/b49f1c8279e9448aa71388cabf69552d.png)
  - 区别：
    - LN中同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；BN中则针对不同神经元输入计算均值和方差，同一个batch中的输入拥有相同的均值和方差
  - 为什么 NLP 使用 LN 而不是 BN？
    - LN 不依赖于 batch 的大小和输入 sequence 的长度，因此可以用于 batchsize 为 1 和 RNN 中 sequence 的 normalize 操作
    - BN 不适用于 batch 中 sequence 长度不一样的情况，有的靠后面的特征的均值和方差不能估算；另外 BN 在 MLP 中的应用对每个特征在 batch 维度求均值方差，比如身高、体重等特征，但是在 NLP 中对应的是每一个单词，但是每个单词表达的特征是不一样的
    - 如果特征依赖于不同样本间的统计参数（比如 CV 领域），那么 BN 更有效(它抹杀了不同特征之间的大小关系，但是保留了不同样本间的大小关系)；NLP 领域 LN 更合适（抹杀了不同样本间的大小关系，但是保留了一个样本内不同特征之间的大小关系），因为对于 NLP 或序列任务来说，一条样本的不同特征其实就是时序上字符取值的变化，样本内的特征关系是非常紧密的
  - 相同点：标准化技术目的是让每一层的分布稳定下来，让后面的层可以在前面层的基础上安心学习，加快模型收敛


#### BN 中的可学习参数量分析
- 对于一个NCHW（批次大小，通道数，高度，宽度）格式的输入张量，Batch Normalization（BN）层通常包含两组可学习参数：缩放因子（scale）和偏移（shift）。
  - 缩放因子（scale）参数：对于每个通道，都有一个缩放因子参数。因此，缩放因子参数的数量等于通道数（C）。
  - 偏移（shift）参数：同样，对于每个通道，都有一个偏移参数。因此，偏移参数的数量也等于通道数（C）。
  - 所以，总共的可学习参数数量等于缩放因子参数和偏移参数的数量之和，即 **2C**。

- 基于 pytorch 的示例 BN 示例代码
```python
# CV Example
import torch
import torch.nn as nn
batch, channel, height, width = 20, 100, 35, 45
input = torch.randn(batch, channel, height, width)
m = nn.BatchNorm2d(channel)
output = m(input)
for k, v in m.named_parameters():
  print(k, v.shape)
# -> weight torch.Size([100])
# -> bias torch.Size([100])
```

#### LN 中的可学习参数量分析

```python
# NLP Example
import torch
import torch.nn as nn
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
output = layer_norm(embedding)
for k, v in layer_norm.named_parameters():
  print(k, v.shape)
# -> weight torch.Size([10])
# -> bias torch.Size([10])
```


#### 优化方法:SGD、Adam、AdamW 对比

- SGD优化阶段的梯度参数量等于模型参数量，Adam在保存优化阶段的梯度参数量外还需要保存其它参数，模型的参数量会在优化区间翻 4 倍
- Adam 的参数是【梯度】的一阶矩和二阶矩，跟网络的【激活值】没啥关系
- 参考： [pytorch中常见优化器的SGD,Adagrad,RMSprop,Adam,AdamW的总结](https://zhuanlan.zhihu.com/p/482768895)
  - Adam(Adaptive Moment Estimation)优化算法，整合了RMSprop中的自适应梯度机制和动量梯度机制。
  - AdamW优化器修正了Adam中权重衰减的bug，Decoupled Weight Decay Regularization
    - AdamW与Adam对比，主要是修改了权重衰减计算的方式，一上来直接修改了，而不是把权重衰减放到梯度里，由梯度更新间接缩小
    - 显存/内存消耗等其他细节和Adam没有区别

## 自然语言处理基础知识
### 分词器(Tokenizer)
#### 分词（Tokenize） 介绍
[参考](https://zhuanlan.zhihu.com/p/448147465)
- 语言模型是对文本进行推理。由于文本是字符串，但对模型来说，输入只能是数字，所以就需要将文本转成用数字来表达。最直接的想法，就是类似查字典，构造一个字典，包含文本中所有出现的词汇，比如中文，可以每个字作为词典的一个元素，构成一个列表；一个句子就可以转换成由每个词的编号（词在词典中的序号）组成的数字表达。分词就是将连续的字序列按照一定的规范重新组合成语义独立词序列的过程，一个分词示例流程如下：
  - Hello Geeks how are you 的一句话分开为每个单词为一个 token： [Hello, Geeks, how, are, you]
  - 执行分词的算法模型称为分词器（Tokenizer） ，划分好的一个个词称为 Token，这个过程称为 Tokenization
  - 由于一篇文本的词往往太多了，为了方便算法模型训练，我们会选取出频率 （也可能是其它的权重）最高的若干个词组成一个词表（Vocabulary）

- 古典分词方法
  - 具体分词方式示例
    - 直接把词与词直接加一个空格
    - 按标点符号分词
    - 语法规则分词
  - 缺点
    - 对于未在词表中出现的词（Out Of Vocabulary, OOV ），模型将无法处理（未知符号标记为 [UNK]）。
    - 词表中的低频词/稀疏词在模型训无法得到训练（因为词表大小有限，太大的话会影响效率）
    - 很多语言难以用空格进行分词，例如英语单词的多形态，"look"衍生出的"looks", "looking", "looked"，其实都是一个意思，但是在词表中却被当作不同的词处理，模型也无法通过 old, older, oldest 之间的关系学到 smart, smarter, smartest 之间的关系。这一方面增加了训练冗余，另一方面也造成了大词汇量问题。

- 拆分为单个字符（Character embedding）
  - 是一种更为极端的分词方法，直接把一个词分成一个一个的字母和特殊符号。虽然能解决 OOV 问题，也避免了大词汇量问题，但缺点也太明显了，粒度太细，训练花费的成本太高

- 基于子词的分词方法（Subword Tokenization）
  - 把一个词切成更小的一块一块的子词，**基于子词的分词方法是目前主流的分词方案**，有点类似英语中的词根词缀拼词法，其中的这些小片段又可以用来构造其他词。可见这样做，既可以降低词表的大小，同时对相近词也能更好地处理
    - 例如：“unfortunately ” = “un ” + “for ” + “tun ” + “ate ” + “ly ”
  - 三种主流的 Subword 算法，它们分别是：Byte Pair Encoding (BPE)、WordPiece 和 Unigram Language Model

#### 字节对编码 Byte Pair Encoding (BPE)、WordPiece、Unigram Language Model 介绍
[参考1](https://zhuanlan.zhihu.com/p/448147465), [参考2](https://zhuanlan.zhihu.com/p/86965595)
- 字节对编码（BPE, Byte Pair Encoder），又称 digram coding 双字母组合编码，是一种数据压缩 算法，用来在固定大小的词表中实现可变⻓度的子词。该算法简单有效，因而目前它是最流行的方法。
  - BPE 首先将词分成单个字符，然后依次用另一个字符替换频率最高的一对字符 ，直到循环次数结束。
  - 算法过程
    1. 准备语料库，确定期望的 subword 词表大小等参数
    2. 通常在每个单词末尾添加后缀 </w>，统计每个单词出现的频率，例如，low 的频率为 5，那么我们将其改写为 "l o w </ w>”：5
    注：停止符 </w> 的意义在于标明 subword 是词后缀。举例来说：st 不加 </w> 可以出现在词首，如 st ar；加了 </w> 表明该子词位于词尾，如 we st</w>，二者意义截然不同
    3. 将语料库中所有单词拆分为单个字符，用所有单个字符建立最初的词典，并统计每个字符的频率，本阶段的 subword 的粒度是字符
    4. 挑出频次最高的符号对 ，比如说 t 和 h 组成的 th，将新字符加入词表，然后将语料中所有该字符对融合（merge），即所有 t 和 h 都变为 th。
    注：新字符依然可以参与后续的 merge，有点类似哈夫曼树，BPE 实际上就是一种贪心算法 。
    5. 重复遍历 2 和 3 操作，直到词表中单词数达到设定量 或下一个最高频数为 1 ，如果已经打到设定量，其余的词汇直接丢弃
  - BPE 一般适用在欧美语言拉丁语系中，因为欧美语言大多是字符形式，涉及前缀、后缀的单词比较多。而中文的汉字一般不用 BPE 进行编码，因为中文是字无法进行拆分。对中文的处理通常只有分词和分字两种。理论上分词效果更好，更好的区别语义。分字效率高、简洁，因为常用的字不过 3000 字，词表更加简短。

- WordPiece：WordPiece算法和BPE类似，区别在于WordPiece是基于概率生成新的subword而不是下一最高频字节对。- Unigram：它和 BPE 等一个不同就是，bpe是初始化一个小词表，然后一个个增加到限定的词汇量，而 Unigram 是先初始一个大词表，接着通过语言模型评估不断减少词表，直到限定词汇量。


#### ChatGPT 的 tokenizer 用于中文分词的问题
[参考](https://zhuanlan.zhihu.com/p/626621158)

- 在 [Openai Tokenizer demo](https://platform.openai.com/tokenizer) 中，中文分词后的 token 数量远大于原始中文字符数目
![openai tokenizer](./2.自然语言处理基础知识/images/openai_token.png)
- 原因剖析：OpenAI 为了支持多种语言的 Tokenizer，采用了文本的一种通用表示：UTF-8 的编码方式，这是一种针对 Unicode 的可变长度字符编码方式，它将一个 Unicode 字符编码为1到4个字节的序列。


#### 为什么 llama 的 tokenizer 需要单独针对中文语料进行词表扩充

### 经典NLP模型
#### Bert

- Bert的预训练主要包含两个任务，MLM和NSP，Masked Language Model任务可以理解为完形填空，随机mask每一个句子中15%的词，用其上下文来做预测；Next Sentence Prediction任务选择一些句子对A与B，其中50%的数据B是A的下一条句子，剩余50%的数据B是语料库中随机选择的，学习其中的相关性。BERT 预训练阶段实际上是将上述两个任务结合起来，同时进行，然后将所有的 Loss 相加
### 困惑度(perplexity)
#### 困惑度 (perplexity) 介绍

## 大语言模型基础知识
### 训练框架介绍(Megatron-lm、DeepSpeed)
#### Megatron-lm 介绍

[参考1](https://arxiv.org/abs/1909.08053), [参考2](https://www.bilibili.com/video/BV1nB4y1R7Yz/?spm_id_from=333.880.my_history.page.click&vd_source=4485ae5d861b0e733e8796b93c824c9e), [参考3](https://huggingface.co/blog/zh/megatron-training)


#### TP 和 PP 通信量对比



#### DDP 具体原理介绍，通信的算子操作具体是什么？



#### DeepSpeed 介绍
### 参数高效微调(PEFT)
#### Pattern-Exploiting Training（PET）
[参考](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw%3D%3D&chksm=96ea6fe7a19de6f1be86b965e268df1b9c6320810cf32b6d64ddd3d238bf9088be41fb36adfe&idx=1&mid=2247512167&scene=21&sn=cc7695d92362e3b18a6e8969fb14dc27#wechat_redirect)
- 它通过人工构建的模版与 BERT 的 MLM 模型结合，能够起到非常好的零样本、小样本乃至半监督学习效果，而且该思路比较优雅漂亮，因为它将预训练任务和下游任务统一起来了

#### P-tuning and P-tuningv2
- P-tuning（GPT Understands, Too）（[参考](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/115648821)）
  - P-tuning 重新审视了关于模版的定义，放弃了“模版由自然语言构成”这一常规要求，从而将模版的构建转化为连续参数优化问题，虽然简单，但却有效
  - P-tuning直接使用[unused*]的token来构建模版，不关心模版的自然语言性
    ![ptuning](./3.大语言模型基础知识/images/ptuning.png)
  - 借助 P-tuning，GPT 在 SuperGLUE 上的成绩首次超过了同等级别的 BERT 模型，这颠覆了一直以来“GPT 不擅长

- P-tuningv2



#### LoRA

[参考1](https://zhuanlan.zhihu.com/p/617211910)，[参考2](https://zhuanlan.zhihu.com/p/643560888#:~:text=%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84%E6%8C%91%E6%88%98-,7.Lora%E7%9A%84%E5%8E%9F%E7%90%86%E5%92%8C%E5%AD%98%E5%9C%A8%E7%9A%84%E9%97%AE%E9%A2%98%E8%AE%B2%E4%B8%80%E4%B8%8B%EF%BC%9F,-%E5%89%8D%E9%9D%A2%E5%9C%A84)
- 低秩自适应 (Low-Rank Adaptation, LoRA)：冻结了预训练的模型权重，并将可训练的秩分解矩阵注入到 Transformer 架构的每一层，极大地减少了下游任务的可训练参数的数量，有效提升预训练模型在下游任务上的 finetune 效率
    ![lora](./3.大语言模型基础知识/images/lora.png)
<!-- <div style="text-align: center;">
    <img src="./3.大语言模型基础知识/images/lora.png" alt="Image Description" width="300">
</div> -->

- 【背景】之前的 PEFT 方法是 adapter/prefix/promp/P-tuning，但是Adapter会引入很强的推理延迟（只能串行），prefix/prompt/P-tuning很难练，而且要占用context length，变相的降低模型能力
- 【详解】具体来说，就是考虑到语言模型（LLM 尤其如此）的参数的低秩属性（low intrinsic dimension），或者说过参数化，在做 finetune 的时候不做 full-finetune，而是用一个降维矩阵A和一个升维矩阵B去做finetune。如果我们认为原来的模型的某个参数矩阵为  $W_{0}$，那么可以认为原来经过全微调的参数矩阵为 $W_{0} + \Delta(W)$ ，但考虑到前面的低秩属性，在 lora 中我们可以简单认为  $\Delta(W)=BA$  (B 是降维矩阵，A是升维矩阵，其中 A 正常随机数初始化，**B 全 0 初始化**，从而保证训练初期的稳定性)，其中 BA 的秩相当于是你认为的模型实际的秩。这样的话在做推理的时候， $h=W_{0}x + BAx$ ，根本不会引入推理延迟，因为你只需要把训好的 lora 参数 $AB$ 加进模型初始权重 $W_{0}$ 中就可以了。在 Transformer 中 self-attention 和 mlp 模块都有对应的 params 矩阵，对应加上 lora 即可。
  - llama 为了节省参数量一般只加在 q、v 上 ([参考](https://github.com/tloen/alpaca-lora/blob/main/finetune.py#L44))，原论文实验是不会掉点
  - bloom 一般加在 q、k、v 上
### 经典开源LLM介绍
#### Llama
- causal LM： 严格遵守只有后面的token才能看到前面的token的规则
- 使用 RoPE 位置编码

#### Llama2

#### ChatGLM and ChatGLM2
- ChatGLM
  - 使用 prefix LM：prefix部分的token互相能看到
  - 使用 RoPE 位置编码
- ChatGLM2
  - 回归 decoder-only 结构，使用 Causal LM
  - 使用 RoPE 位置编码

#### Bloom
- BLOOM 使用 Alibi 位置编码

### RLHF介绍
#### PPO (Proximal Policy Optimization) 介绍
[参考1](https://huggingface.co/blog/deep-rl-ppo), [参考2](https://zhuanlan.zhihu.com/p/645225982)

- 大语言模型 RLHF 中的 PPO 分成三部分：采样、反馈和学习
  - 采样就是学生回答问题的过程，是模型根据提示（prompt）输出回答（response）的过程，或者说是模型自行生产训练数据的过程
    - 策略（policy），它就是RLHF中的“学生”。policy由两个模型组成，一个叫做演员模型（Actor），另一个叫做评论家模型（Critic）。它们就像是学生大脑中的两种意识，一个负责决策，一个负责总结得失。其中演员就是我们想要训练出来的大模型。在用PPO训练它之前，它就是RLHF的第一步训练出来的SFT（Supervised Fine-Tuning）model。
  - 反馈就是老师检查答案的过程，是奖励模型（Reward Model）给response打分的过程，或者说是奖励模型给训练数据X标上Y值的过程
    - 给予最终奖励之前，最好也对它的“标新立异”给予少量的惩罚。实现方式是让 old_log_prob 的 token 概率分布尽量接近 ref_log_prob。通俗来说，整个reward function的计算逻辑是典型的霸总逻辑：除非你能拿到好的结果，否则你就得给我守规矩。
  - “学习“就是学生根据反馈总结得失并自我改进的过程，或者说是强化优势动作的过程。
    - 优势定义为“实际获得的收益超出预期的程度”。PPO计算优势的方法：优势 = 实际收益 - 预期收益。
    - 所谓“强化优势动作”，即强化那些展现出显著优势的动作。

#### DPO (Direct Preference Optimization) 介绍
[参考1](https://arxiv.org/pdf/2305.18290.pdf)，[参考2](https://zhuanlan.zhihu.com/p/636122434)
- 背景
  - 目前常用人类对模型生成数据的反馈来进一步训练LM，对齐人类偏好；RLHF是其中的代表工作，先通过训练一个反映人类对生成回答偏好的reward model（RW），再通过强化学习（PPO）来最大化预测的回报（reward），同时，施加KL限制（constraints）避免模型偏离太远。
  -这个 pipeline 比较复杂，需要先训练一个reward model 得到回报分数，再通过 PPO 强化学习最大化 reward 更新策略（模型参数），其中的 PPO 阶段，需要多个模型（actor&ref model& critic model& reward model），特别耗显存，并且在训练过程中，需要对策略进行采样，计算量巨大


- DPO 实现方式
  ![dpo pipeline](./3.大语言模型基础知识/images/dpo.png)
  - 直接优化 LM 来对齐人类偏好，无需建模 reward model 和强化学习阶段。基于 RL 的目标函数可以通过优化二分 cross entropy 目标来优化
  - DPO loss 表达为如下形式
  ![dpo loss](./3.大语言模型基础知识/images/dpo_1.png)
  其中 $y_w$ 的被人类喜好程度大于 $y_l$，DPO loss 的梯度如下
  ![dpo gradient](./3.大语言模型基础知识/images/dpo_2.png)
  可以看出该 loss 的作用主要是增加喜好数据的 likelihood，降低非喜好数据的 likelihood，同时会基于隐私 reward 估计的错误程度进行加权。本文的实验表明了这种加权的重要性，因为没有加权系数的这种方法的简单版本可能会导致语言模型退化

### CoT、ToT、GoT介绍
#### 思维链 CoT (Chain-of-Thought) 介绍

### SFT训练
#### SFT 训练如何提升训练效率


#### 多轮对话数据如何计算损失
## 大语言模型推理
### Huggingface推理参数介绍
#### 温度系数（Temperature）对大模型推理结果的影响
参考：[What is Temperature in NLP / LLMs?](https://medium.com/@lazyprogrammerofficial/what-is-temperature-in-nlp-llms-aa2a7212e687), [NLP / LLMs中的Temperature 是什么?](https://blog.csdn.net/deephub/article/details/129682591)
- 温度系数直观解释
  - Temperature 是一个超参数，可用于控制生成语言模型中生成文本的随机性和创造性。它用于调整模型的softmax输出层中预测词的概率。温度参数定义为在应用 softmax 函数之前用于调整 logits 的比例因子的倒数。
  - 当Temperature 设置为较低的值时，预测词的概率会变尖锐，这意味着选择最有可能的词的概率更高。这会产生更保守和可预测的文本，因为模型不太可能生成意想不到或不寻常的词。另一方面，当Temperature 设置为较高值时，预测词的概率被拉平，这意味着所有词被选择的可能性更大。这会产生更有创意和多样化的文本，因为模型更有可能生成不寻常或意想不到的词。
  - 温度参数通常设置为 0.1 到 1.0 之间的值，具体取决于生成文本中所需的随机性和创造性水平。温度值为 1.0 对应于标准 softmax 函数，其中预测词的概率未按比例缩放。

- 温度系数数学解释
  - LLM 输出为每个单词的预测概率分布，由 softmax 函数计算
    $p(x)=\frac{e^{x_{i}}}{\sum_{N}^{j=1}{e^{x_{j}}}}$
  - 加上温度系数后的 softmax 函数形式为：
    $p(x)=\frac{e^{\frac{x_{i}}{T}}}{\sum_{N}^{j=1}{{e^{\frac{x_{j}}{T}}}}}$
    - 如果当T趋于无穷时会发生什么。每个x_i / T都会趋于0，从而得到一个均匀分布。也就是说概率分布变得更 “平”， 这会导致结果更随机。
    - 当T很小(比如0.1)时会发生什么。每个x_i / T之间的差异变得更加明显(例如5比1变成50比10)，这样概率分布变得“更尖”，也就是说结果会更确定。

#### Huggingface generate 函数中的 top_p sampling、top_k sampling、greedy_search、beam_search 参数解释
参考：[Huggingface 的 generate 方法介绍：top_p sampling、top_k sampling、greedy_search、beam_search](https://zhuanlan.zhihu.com/p/643949567)
- Greedy search 每次都选择概率最大的词作为单词序列中的下一个词
- beam search（束搜索）：通过在每个时间步骤保留最有可能的 num_beams 个假设，最终选择具有最高概率的假设，从而降低错过隐藏的高概率词序列的风险
- top-k sampling：在Top-K采样中，选择最有可能的 K 个下一个词，并将概率质量重新分配给这 K 个下一个词
- Top-p (nucleus) sampling：与仅从最可能的 K 个单词中进行采样不同，Top-p 采样从概率累积超过概率 p 的可能性最小的单词集中进行选择


#### Huggingface generate 生成文本重复有什么参数调节
[参考1](https://huggingface.co/blog/how-to-generate), [参考2](https://arxiv.org/pdf/1909.05858.pdf), [参考3](https://arxiv.org/abs/1705.04304)

基于以下参数能避免重复样本生成

- repetition_penalty
  - 文档：[LINK](https://huggingface.co/transformers/v4.1.1/internal/generation_utils.html#transformers.RepetitionPenaltyLogitsProcessor)
  - 源码：[LINK](https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html)
  - 参数作用：对重复序列实施惩罚，设置为 1.0 代表没有惩罚，一般设置为 1.1 或 1.2 就能缓解重复序列的生成
  - 实现原理：在语言模型输出的 logits 通过乘以设置的 `repetition_penalty` 来降低重复序列的权重，具体代码如下，对于小于 0 的部分乘以 penalty 参数，大于 0 的部分乘以 penalty 参数，这里 penalty 一般设置为 1~2 之间，所以能够降低重复序列的值。核心代码如下：
    ```python
    score = torch.where(score < 0, score * self.penalty, score / self.penalty)
    ```

- no_repeat_ngram_size
  - 文档：[LINK](https://huggingface.co/transformers/v4.1.1/internal/generation_utils.html#transformers.NoRepeatNGramLogitsProcessor)
  - 源码：[LINK](https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html)
  - 参数作用：
    - 对于包含了相同的单词序列的重复，设置该参数可以引入[论文](https://arxiv.org/abs/1705.04304)中介绍的 n-gram (即 n 个单词的单词序列) 对重复单词序列进行惩罚，确保不会出现重复的 n-gram 。设置 `no_repeat_ngram_size=n` 能够促使 n-gram 不会重复出现
  - 实现原理：基于输入的文本中统计已经出现的单词序列，将这部分单词序列的出现概率置 0。核心代码如下：
    ```python
    for i, banned_tokens in enumerate(banned_batch_tokens):
      scores[i, banned_tokens] = -float("inf")
    ```
    注意这里将 `logits` 置为负无穷后，经过 softmax 函数后就实现将概率置 0 了。

## 大语言模型应用
### LangChain介绍
#### LangChain 介绍
参考：[大语言模型集成工具 LangChain](https://zhuanlan.zhihu.com/p/599688026)
- LangChain 通过可组合性使用大型语言模型构建应用程序
- 【背景】大型语言模型 (LLM) 正在成为一种变革性技术，使开发人员能够构建他们以前无法构建的应用程序，但是单独使用这些 LLM 往往不足以创建一个真正强大的应用程序，当可以将它们与其他计算或知识来源相结合时，就有真的价值了。LangChain 旨在协助开发这些类型的应用程序
- LangChain 主要功能：
  - 实现统一接口，支持不同大语言模型的统一化调用
  - 支持引入 google 搜索、python 解释器等外部工具
  - 支持便捷 prompt 模板设置（并提供一些教科书 prompt 模板供参考）
  - 支持智能体 (Agent) 等高阶应用
## 大语言模型前沿分享
### LLM相关博客分享
#### LLM 博客
- [State of GPT (OpenAI Karpathy 介绍 ChatGPT 原理及现状)](https://zhuanlan.zhihu.com/p/634038281)
  - Andrej Karpathy 介绍如何训练 ChatGPT 以及如何将 ChatGPT 用于定制化应用程序
- [A Stage Review of Instruction Tuning](https://yaofu.notion.site/June-2023-A-Stage-Review-of-Instruction-Tuning-f59dbfc36e2d4e12a33443bd6b2012c2)
  - 符尧讲解 SFT 现状及可以关注的问题
- [C-Eval: 构造中文大模型的知识评估基准](https://yaofu.notion.site/C-Eval-6b79edd91b454e3d8ea41c59ea2af873)
  - 介绍构造 C-Eval 的过程，同时介绍了提升模型排名的方法
### LLM相关论文分享
#### 大模型涌现分析
- [大模型涌现能力探讨：大型语言模型的涌现能力是幻象吗？](https://zhuanlan.zhihu.com/p/634039621)
  - 探索大语言模型表现出涌现能力的原因，初步结论是涌现能力主要是由研究人员选择一个非线性或不连续的评价指标导致的，另外探索了如何诱导涌现能力的出现，本文在视觉任务上通过对评价指标的修改复现了涌现现象。
#### 大语言模型知识蒸馏

- [Fine-tune-CoT](https://zhuanlan.zhihu.com/p/597395236)： 旨在利用非常大的语言模型 (LMs) 的CoT推理能力来教导小模型如何解决复杂任务，蒸馏出来的小模型在某些数据集上精度甚至能超过 teacher 大模型
