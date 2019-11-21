---
layout: post
title: copy(pointer) making the seq2seq with attention better
categories: [NLP] 
---

本篇post主要介绍如何在基于attention机制的seq2seq(Encoder-Decoder)框架中，进一步引入copy机制，使得解码输出更加流畅与准确。

这里主要总结两篇关于copy机制的paper：[Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)与[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)。

## Incorporating Copying Mechanism in Sequence-to-Sequence Learning

该paper中提出的copyNet使得解码器在解码时能自行决定预测的token是由生成模式还是复制模式而来。其网络结构主要是在[Bahdanau attention](https://arxiv.org/abs/1409.0473)的基础上进一步增加了copy机制。该机制对于摘要和对话系统来说，可有效提高基于端到端生成文本的流畅度和准确度，并改善了OOV问题。

### RNN Encoder-Decoder with attention
首先回顾下基于Bahdanau attention的Encoder-Decoder的过程。

在**Encoder端**，输入序列$$X=\left[ x_1,\cdots ,x_{T_s} \right]$$被编码成相应的隐状态(向量)序列$$h_1,\cdots ,h_{T_s}$$，其中$$h_t=f\left( x_t,h_{t-1} \right)$$，**$$h_0$$为随机初始化的编码器端的隐状态。**

在**Decoder端**，首先根据上一时刻解码器的输出$$y_{t-1}$$、隐状态$$s_{t-1}$$以及当前时刻由编码器生成的隐状态序列对应的context向量$$c_t$$，更新当前时刻解码器的隐状态$$s_t$$。即

$$
s_t=f\left( y_{t-1},s_{t-1},c_t \right)
$$

然后根据当前时刻解码器的隐状态$$s_t$$、上一时刻解码器的输出$$y_{t-1}$$以及当前时刻由编码器生成的隐状态序列对应的context向量$$c_t$$，预测当前时刻解码器的输出$$y_{t}$$，即

$$
p\left( y_t\left| y_{<t},X \right. \right) =g\left( y_{t-1},s_t,c_t \right) 
$$

其中**$$s_0$$为随机初始化的解码器端的隐状态，$$y_0$$为一种类\<START\>的特殊的字符，表示一个句子的开始，** $$c_t=\sum_{\tau =1}^{T_s}{\alpha _{t\tau}h_{\tau}}$$表示对encoder端的隐状态(向量)序列进行加权求和，

$$\alpha _{t\tau}=\frac{e^{\eta \left( s_{t-1},h_{\tau} \right)}}{\sum_{\tau ^{'}}{e^{\eta \left( s_{t-1},\tau ^{'} \right)}}}$$

表示encoder端各个隐状态(向量)序列的权重，**$$\eta$$表示解码器端上一时刻的隐状态$$s_{t-1}$$与编码器端的某个隐状态(向量)间的相关性得分，这使得在更新当前时刻解码器的隐状态$$s_t$$前，需先计算context向量$$c_t$$。**

**在上述Decoder时，相关变量的迭代顺序：先计算context向量$$c_t$$，接着更新Decoder的隐状态$$s_t$$，最后估算最有可能的$$y_t$$，如此循环。**

最终针对一个样本(源序列和目标序列)，其目标为最小化如下似然函数：

$$
\mathcal{L}=-\sum_{t=1}^T{\log \left[ p\left( y_t\left| y_{<t},X \right. \right) \right]}
$$

### copyNet
从认知角度看，copy机制就是一种不需要理解内在语义，只需确保文字的外在准确性的一种死记硬背的形式。从模型角度看，copy操作更加生硬，相比上述的注意力机制，其更难集成到一个完全可微的神经网络中。作者们给出了一种基于copy机制且可微的seq2seq模型结构，整体结构如下图所示：

![png](/assets/images/nlp/seq2seq/copyNet-01.png)

与上述分析的基于attention的Encoder-Decoder相比，最大不同在于Decoder部分，Encoder部分不变。值得指出的是，**作者将Encoder端生成的隐状态向量序列定义为copyNet的短期memory，用$$M$$表示。在解码时$$M$$将被多次使用。** 

下面重点对比下copyNet模型中的Decoder部分与上述Decoder部分的不同点。

- **Prediction：** copyNet在预测某个token时是基于生成模式和copy模式两种混合概率进行建模的，即

$$
p\left( y_t\left| s_t,y_{t-1},c_t,M \right. \right) =p\left( y_t,g\left| s_t,y_{t-1},c_t,M \right. \right) +p\left( y_t,c\left| s_t,y_{t-1},c_t,M \right. \right) 
$$

- **State Update：** 在RNN Encoder-Decoder with Bahdanau attention中，当前时刻解码器的状态$$s_t$$根据上一时刻解码器的输出$$y_{t-1}$$、隐状态$$s_{t-1}$$以及当前时刻由编码器生成的隐状态序列对应的context向量$$c_t$$进行更新；而在copyNet中，将$$y_{t-1}$$替换成$$y_{t-1}$$的embedding$$e\left( y_{t-1} \right)$$与$$y_{t-1}$$在输入序列中对应的隐状态向量(**若decoder在上一时刻的输出$$y_{t-1}$$不在encoder的输入序列$$X=\left[ x_1,\cdots ,x_{T_s} \right]$$中，则对应的隐状态向量为零向量**)$$\zeta \left( y_{t-1} \right) $$的拼接$$\left[ e\left( y_{t-1} \right) ;\zeta \left( y_{t-1} \right) \right] ^T$$(**此项体现了copy的思想**)，其它项不变，即

$$
s_t=f\left( \left[ e\left( y_{t-1} \right) ;\zeta \left( y_{t-1} \right) \right] ^T,s_{t-1},c_t \right) 
$$

- **Reading M：** 在解码端更新当前时刻的隐状态时，除了像上述attention机制，**每次动态地将encoder端生成的所有隐状态向量序列($$M$$)表示成(加权求和，attentive read)该时刻对应的context向量**外，还会检查解码器上一时刻的输出$$y_{t-1}$$在输入序列$$X$$中的位置(**同一个词可能会出现多次**)，然后在$$M$$中**取出(selective  read)相应位置对应的隐状态向量$$\zeta \left( y_{t-1} \right) $$**，不在的话，对应隐状态为零向量。 **$$M$$的两种读取方式(attentive read与selective  read，本质上都是基于注意力进行相关运算的)使得copyNet可以在Generate-Mode与Copy-Mode间进行切换，甚至决定何时开始或者结束copy。Attentive Read就是在编码器端的attention mechanism，Selective Read就是 $$\zeta \left( y_{t-1} \right) $$的计算过程。如果$$y_{t-1}$$在输入序列$$X$$中,那么copyNet接下去的输出就很可能偏向Copy-Mode。**

上述为copyNet内部结构的大致描述，有关具体变量的详细描述见[paper](https://arxiv.org/abs/1603.06393)，不再赘述。其整体结构和相关计算与RNN Encoder-Decoder with attention部分很相似，主要不同点在于Decoder部分。

网上找的关于CopyNet的实现。

- tensorflow: [CopyNet Implementation with Tensorflow and nmt](https://github.com/lspvic/CopyNet)

- pyTorch: [An implementation of "Incorporating copying mechanism in sequence-to-sequence learning"](https://github.com/mjc92/CopyNet)

- Theano: **paper作者** [incorporating copying mechanism in sequence-to-sequence learning](https://github.com/MultiPath/CopyNet)

- paper中用的文本摘要的公开数据集LCSTS(Large Scale Chinese Short Text Summarization)：[https://pan.baidu.com/s/1rJC9Vk8e3gF38-mAI5JHPw](https://pan.baidu.com/s/1rJC9Vk8e3gF38-mAI5JHPw)


## Get To The Point: Summarization with Pointer-Generator Networks
该篇paper针对现有基于attention的seq2seq模型做生成式文本摘要的缺点：1、模型可能会错误地生成事实细节；2、模型可能会重复生成某些内容。作者们在原有模型基础上提出了两种改进的方案：1、解码时使用混合的pointer-generator网络结构(**思想与copyNet相同，形式不同，相比copyNet，结构更加清晰**)。即利用pointer从源文本中copy内容，可有效避免错误事实的生成；利用generator从词表中生成词。pointer-generator结构使模型同时具备了抽取式和生成式摘要的能力； 2、解码输出时增加coverage约束。即在解码时抑制已经在源文本中出现过的内容的attention，可有效消除重复内容的出现。基于改进后的模型在文本摘要任务上可取得较好效果。

### Sequence-to-sequence attentional model
这部分内容的计算流程和上述RNN Encoder-Decoder with attention部分大致相同，只是符号有所区别。为了方便与改进后的模型进行对比，给出改进前的模型图：

![png](/assets/images/nlp/seq2seq/pointer-generator-01.png)

图中在$$t$$时刻进行解码时有**两个概率分布**：attention分布$$a^{t}$$和词表分布$$P_{vocab}$$。

其中$$a^t=softmax \left( e^t \right) $$表示$$t$$时刻编码器端每个隐状态向量对应的权重值，且$$e_{i}^{t}=v^T\tan\text{h}\left( W_hh_i+W_ss_{t-1}+b_{attn} \right)$$表示解码器端$$t-1$$时刻的隐状态向量$$s_{t-1}$$(**paper中写成了t，个人觉得有问题，和Bahdanau不一致**)与编码器端第$$i$$个隐状态向量$$h_i$$的相关性得分。

由词表分布可将生成词$$w$$的概率表示为$$P\left( w \right) =P_{vocab}\left( w \right)$$。

训练期间，在$$t$$时刻预测目标词$$w_{t}^{*}$$对应的loss为：

$$loss_t=-\log P\left( w_{t}^{*} \right) $$

### Pointer-generator network
改进后的模型图：

![png](/assets/images/nlp/seq2seq/pointer-generator-02.png)

与copyNet类似，**Pointer-generator network在解码端预测某个词$$w$$的概率时，也综合考虑从词表生成(词表分布$$P_{vocab}$$)和从源输入复制(attention分布)两种情况。** 不过其表现形式不同，具体如下：

$$
P\left( w \right) =p_{gen}P_{vocab}\left( w \right) +\left( 1-p_{gen} \right) \sum_{i:w_i=w}{a_{i}^{t}}
$$

其中$$p_{gen}=sigmoid\left( w_{h^*}^{T}h_{t}^{*}+w_{s}^{T}s_t+w_{x}^{T}x_t+b_{ptr} \right)$$表示生成模式的概率，$$h_{t}^{*}$$表示$$t$$时刻计算的context向量，$$s_t$$表示解码器的隐状态，$$x_t$$表示解码器的输入，$$w_{h^*}、w_{s}、w_{x}以及b_{ptr}$$为可学习的模型参数。

**说明：** 如果词$$w$$不在词表中，即OOV，则$$P_{vocab}\left( w \right) = 0$$。如果词$$w$$不在源输入序列中，则$$\sum_{i:w_i=w}{a_{i}^{t}}=0$$。

### Coverage mechanism
在生成多句文本序列时，很容易出现内容重复现象。为了消除这种现象，作者们提出了一种coverage机制。具体地，在每一时刻会维持一个coverage向量$$c^t$$，其是**解码端在之前的各个时刻的attention分布之和**，即

$$c^t=\sum_{t^{'}=0}^{t-1}{a^{t^{'}}}$$

其表示到目前$$t$$时刻为止，源文本输入序列在attention机制下覆盖密度的分布(未归一化)情况。其中$$c^0$$为零向量(**初始阶段没有覆盖源文本序列的任何词**)。

考虑Coverage机制后，在计算解码器端$$t-1$$时刻的隐状态向量$$s_{t-1}$$(**paper中写成了t，个人觉得有问题，和Bahdanau不一致**)与编码器端第$$i$$个隐状态向量$$h_i$$的相关性得分时，需要同时考虑上述coverage向量。即

$$
e_{i}^{t}=v^T\tan\text{h}\left( W_hh_i+W_ss_{t-1}+w_cc_{i}^{t}+b_{attn} \right) 
$$

这保证了当前的相关性得分参考了之前的权重累积和，从而有效抑制重复内容的出现。同时作者们又定义了一个额外的coverage损失来进一步惩罚在copy时相同位置copy多次的情况。具体损失函数为：

$$
covloss_t=\sum_i{\min \left( a_{i}^{t},c_{i}^{t} \right)}
$$

其中$$covloss_t\le 1$$。值得指出的是，上述损失函数仅针对每个注意分布和覆盖范围间的重叠进行惩罚，**防止重复关注。**

基于上述coverage损失，最终加上coverage机制后，训练期间在$$t$$时刻预测目标词$$w_{t}^{*}$$时的loss为

$$
loss_t=-\log P\left( w_{t}^{*} \right) +\lambda \sum_i{\min \left( a_{i}^{t},c_{i}^{t} \right)}
$$

以上便是作者们提出的pointer-generator network + coverage mechanism的思路流程，同时作者也开源了整体模型的实现：[Code for the ACL 2017 paper "Get To The Point: Summarization with Pointer-Generator Networks"](https://github.com/abisee/pointer-generator)

## 参考

- [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393)

- Bahdanau attention: [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)

- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)

## attention 扩展

- [https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)

- [遍地开花的 Attention ，你真的懂吗？](https://mp.weixin.qq.com/s/MzHmvbwxFCaFjmMkjfjeSg)