---
layout: post
title: 机器学习中常见优化算法总结
categories: ML
---
机器学习的过程通常是一种利用**某种优化算法**使**模型的目标函数****最小化或最大化**的过程。以下总结机器学习过程中常用的优化算法。

## 梯度下降(gradient descent)
给定某个目标函数(损失函数)$$f\left( \text{x} \right)$$，沿着**梯度的反方向$$-\nabla f\left( \text{x} \right) $$**以**一定(过大的学习率目标函数值可能会发散)学习率$$\eta$$**更新自变量，即

$$\text{x}\gets \text{x}-\eta \nabla f\left( \text{x} \right) $$

可减小自变量对应的目标函数的值。不断迭代上述过程，当自变量$$\text{x}$$对应的梯度(导数)$$\nabla f\left( \text{x} \right) $$趋向于0时，其对应的目标函数值$$f\left( \text{x} \right)$$可取得较小值(当目标函数为凸函数时为最小值，否则为局部极小值)。此时的自变量$$\text{x}$$为目标函数的解。

由上述分析可知，可直接近似求解目标函数(损失函数)$$f\left( \text{x} \right)$$的梯度为0时对应的解。即求$$\nabla f\left( \text{x} \right) =0$$的解。通过**牛顿法**可近似求解上述方程。具体迭代公式如下：

$$\text{x}_{n+1}=\text{x}_n-\frac{f^{'}\left( \text{x}_n \right)}{f^{''}\left( \text{x}_n \right)}$$

随机选取一点，由上述迭代公式不断迭代，直到自变量收敛，**相关超参数只有学习率**。
## 随机梯度下降
与**梯度下降每次更新自变量时取所有样本**不同的时，**随机梯度下降每次更新自变量时只取一个样本**，**相关超参数只有学习率**。

## 小批量随机梯度下降
小批量随机梯度下降每次更新自变量时取$$minibatch$$个样本，**相关超参数为学习率和$$minibatch$$**。

## 动量(momentum)

上述关于梯度下降的各种算法在计算关于自变量的梯度时**仅考虑了自变量当前位置的梯度**，当样本数据中存在噪声时可能会带来一些问题。当**目标函数的二阶矩阵**对应的**最大特征值与最小特征值之比**过大时，**过大的学习率**会使**目标函数**在迭代过程中**在某个方向上发散**，**过小的学习率**会使**目标函数**在迭代过程中**在某个方向上收敛的过慢**。为了解决上述问题，有两种方案：

- 预处理梯度向量：如在二阶优化时利用Hessian阵的逆对梯度进行变换，即$$\text{x}\gets \text{x}-H^{-1}g$$。应用在Adam、RMSProp、AdaGrad、AdaDelta和其他二阶优化算法中
- 平均历史梯度：对多个位置的梯度进行滑动平均，如动量，**允许使用较大的学习率来加速收敛**。应用在Adam、RMSProp、SGD momentum

### 指数加权移动平均
为了从数学上理解动量法，首先介绍下指数加权移动平均(exponential moving average)。

给定超参数$$0\leqslant \beta <1$$，当前时间步$$t$$的变量$$y_t$$是上一时间步$$t-1$$的变量$$y_{t-1}$$和当前时间步另一变量$$x_t$$的线性组合，即

$$
y_t=\beta y_{t-1}+\left( 1-\beta \right) x_t
$$

对上式迭代，可得
$$ 
y_t=\left( 1-\beta \right) x_t+\beta y_{t-1}
\\
=\left( 1-\beta \right) x_t+\beta \left( \left( 1-\beta \right) x_{t-1}+\beta y_{t-2} \right) 
\\
=\left( 1-\beta \right) x_t+\left( 1-\beta \right) \cdot \beta x_{t-1}+\beta ^2y_{t-2}
\\
=\left( 1-\beta \right) x_t+\left( 1-\beta \right) \cdot \beta x_{t-1}+\beta ^2\left( \left( 1-\beta \right) x_{t-2}+\beta y_{t-3} \right) 
\\
=\left( 1-\beta \right) x_t+\left( 1-\beta \right) \cdot \beta x_{t-1}+\left( 1-\beta \right) \cdot \beta ^2x_{t-2}+\beta ^3y_{t-3}
\\
=\left( 1-\beta \right) \sum_{i=0}^{t-1}{\beta ^ix_{t-i}}+\beta ^ty_0
$$

又
$$
\sum_{i=0}^{t-1}{\beta ^i=}\frac{1-\beta ^t}{1-\beta}
$$

将$$y_t$$的初始时刻的值初始化为0，即$$y_t = 0$$，那么$$y_t$$可看作是对最近$$\frac{1}{1-\beta}$$个时间步的$$x_t$$值的加权平均。当$$\beta=0.9$$时，$$y_t$$可看作是对最近10个时间步的$$x_t$$值的加权平均，且离当前时间步$$t$$越近的$$x_t$$值，其获得的权重越大。

有了指数加权移动平均的背景后，理解**动量算法**就更明朗些。具体地，设时间步$$t$$的自变量为$$x_t$$，学习率为$$\eta _t$$，初始时刻的动量变量$$m_t=0$$，在时间步$$t>0$$，动量法按如下公式进行迭代：

$$
m_t\gets \beta m_{t-1}+\left( 1-\beta \right) g_t
\\
x_t\gets x_{t-1}-\alpha _tm_t
\\
\alpha _t=\frac{\eta _t}{1-\beta}
$$

其中**动量超参数$$\beta$$**满足$$0\leqslant \beta <1$$。由上述指数加权移动平均的分析可知，**动量变量$$m_t$$实际上是对序列$$\left\{ g_t:t=0,1,\cdots ,\frac{1}{1-\beta}-1 \right\} $$做了指数加权移动平均。所以在动量法中，自变量在各个方向上的更新不仅取决于当前梯度值，还取决于过去$$\frac{1}{1-\beta}$$个时刻的梯度值。** 且当$$\beta=0$$时，动量法等价于梯度下降。在pyTorch中，torch.optim.SGD已实现了Momentum，其在原有学习率的基础上增加了动量的超参数，**相关超参数**为{'lr': 0.004, 'beta': 0.9}。

## AdaGrad
在上述梯度和动量的优化算法中，目标函数**自变量的每一个分量**在相同的时间步都使用**同一个学习率**来自我迭代。而**当自变量的分量间的梯度有较大差别时，需要选取足够小的学习率使得目标函数值在梯度值较大的分量上不发散，但在梯度值较小的分量上过小的学习率会导致目标函数值收敛的较慢**。动量法通过指数加权移动平均使得自变量的更新方向更加一致，从而降低较大的学习率会使目标函数值发散的概率。

**AdaGrad根据自变量的每个分量的梯度值的大小来自动调整各个分量上的学习率**，从而避免在梯度和动量的优化算法中统一的学习率难以适应所有维度的问题。其利用一个对小批量随机梯度$$g_t$$按元素平方的累加变量$$s_t$$来自动调整各分量上的学习率。

具体地，初始时刻$$s_0 = 0$$，在时间步$$t$$，首先将小批量随机梯度$$g_t$$按元素平方后累加到变量$$s_t$$：

$$
s_t\gets s_{t-1}+g_t\odot g_t
$$

接着目标函数的自变量的每个分量按如下形式迭代：

$$
x_t\gets x_{t-1}-\frac{\eta}{\sqrt{s_t+\epsilon}}\odot g_t
$$

其中$$\epsilon$$是为了保证数值的稳定性而添加的非常小的常数。

上述的开方、除法和乘法都是按元素运算的，这使得**目标函数的自变量在迭代更新时每个分量分别拥有自己的学习率。** 由于梯度按元素平方的累加变量$$s_t$$在学习率的分母项中，如果目标函数关于自变量的某个分量的偏导数一直都较大，则该分量的学习率将下降较快；反之，如果某个分量的偏导数一直都较小，那么该分量的学习率将下降较慢。**由于$$s_t$$一直对梯度变量$$g_t$$按元素的平方进行累加，自变量的每个分量的学习率在迭代过程中一直在降低或不变。 因此在迭代早期如果解依然不理想，在后期由于学习率过小，可能较难找到一个有用的解。** 在pyTorch中，torch.optim.Adagrad已实现了AdaGrad优化算法，**相关超参数只有学习率**。


## RMSProp
为了解决AdaGrad算法在调整学习率时的问题，RMSProp对其进行了修改。不同于AdaGrad算法中的状态变量$$s_t$$是截至时间步$$t$$所有小批量随机梯度$$g_t$$按元素的平方在自变量的各个维度上分别**求和**，RMSProp将梯度$$g_t$$按元素的平方在自变量的各个维度上分别做**指数加权移动平均**。具体地：

$$
v_t\gets \beta v_{t-1}+\left( 1-\beta \right) g_t\odot g_t
$$

接着和AdaGrad算法一样，也是将目标函数自变量中每个分量的学习率按不同分量进行调整后更新自变量：

$$
x_t\gets x_{t-1}-\frac{\alpha}{\sqrt{v_t+\epsilon}}\odot g_t
$$

由于RMSProp的状态变量$$v_t$$是对平方项$$g_t\odot g_t$$的指数加权移动平均，可看作是最近$$\frac{1}{1-\beta}$$个时间步对平方项的加权平均。这样**自变量每个分量的学习率在迭代过程中就不再一直降低或不变。** 在pyTorch中，torch.optim.RMSprop已实现RMSProp。**相关超参数为{'lr': 0.01, 'beta': 0.9}**。

## AdaDelta
除了RMSProp算法外，AdaDelta也针对AdaGrad在迭代后期可能较难找到有用解的问题做了改进。**AdaDelta中没有学习率这一超参数**。

AdaDelta和RMSProp一样，使用状态变量$$s_t$$保存梯度$$g_t$$按元素平方的指数加权移动平均变量，即

$$
s_t\gets \rho s_{t-1}+\left( 1-\rho \right) g_t\odot g_t
$$

同时，AdaDelta还维护一个额外的状态变量$$\varDelta x_t$$，其初始时间步的值为0，即$$\varDelta x_0 = 0$$，并利用$$\varDelta x_{t-1}$$更新自变量的梯度值，具体地：

$$
g_{t}^{'}\gets \frac{\sqrt{\varDelta x_{t-1}+\epsilon}}{\sqrt{s_t+\epsilon}}\odot g_t
$$

其中$$\epsilon$$是为了保证数值的稳定性而添加的常数。接着更新自变量：

$$
x_t\gets x_{t-1}-g_{t}^{'}
$$

最后，使用$$\varDelta x_t$$来记录更新后的自变量的梯度值$$g_{t}^{'}$$按元素平方的指数加权移动平均：

$$
\varDelta x_t\gets \rho \varDelta x_{t-1}+\left( 1-\rho \right) g_{t}^{'}\odot g_{t}^{'}
$$

不考虑$$\epsilon$$的影响，AdaDelta与RMSProp的不同之处在于使用$$\sqrt{\varDelta x_{t-1}}$$来代替超参数$$\eta$$。在pyTorch中，torch.optim.Adadelta实现了Adadelta，**相关超参数**为{'rho': 0.9}。

## Adam
Adam综合了动量(momentum)与RMSProp，分别引入了动量变量$$m_t$$(**记录了梯度$$g_t$$的指数加权移动平均**)和速度变量$$v_t$$(**记录了梯度$$g_t$$按元素平方$$g_t\odot g_t$$的指数加权移动平均**)。其初始值均为0，即$$m_0=0，v_0=0$$。对于时间步$$t$$，其迭代公式分别为：

$$
m_t\gets \beta _1m_{t-1}+\left( 1-\beta _1 \right) g_t
$$

$$
v_t\gets \beta _2v_{t-1}+\left( 1-\beta _2 \right) g_t\odot g_t
$$

其中$$0\leqslant \beta _1<1$$(**算法作者建议为0.9**)，$$0\leqslant \beta _2<1$$(**算法作者建议为0.999**)。

由指数加权移动平均部分的数学分析可知，$$m_t$$与$$v_t$$可进一步化简为$$m_t=\left( 1-\beta _1 \right) \sum_{i=0}^{t-1}{\beta _{1}^{i}g_{t-i}}$$与$$v_t=\left( 1-\beta _2 \right) \sum_{i=0}^{t-1}{\beta _{2}^{i}g_{t-i}\odot g_{t-i}}$$。将过去各个时间步的梯度的**权值相加**，即$$\left( 1-\beta _1 \right) \sum_{i=0}^{t-1}{\beta _{1}^{i}}=1-\beta _{1}^{t}$$。**值得注意的是，当$$t$$较小时，过去各时间步的梯度对应的权重之和会较小。如当$$\beta _1=0.9$$时，$$m_1=0.1g_t$$。为了消除这样的影响，对任意时间步$$t$$，将$$m_t$$再除以$$1-\beta _{1}^{t}$$，从而使过去各时间步的梯度对应的权重之和为1，该变换也叫偏差修正。** 修正后的动量变量$$\hat{m}_t$$和速度变量$$\hat{v}_t$$分别为：

$$
\hat{m}_t\gets \frac{m_t}{1-\beta _{1}^{t}}
\\
\hat{v}_t\gets \frac{v_t}{1-\beta _{2}^{t}}
$$

利用上述修正后的状态变量$$\hat{m}_t$$与$$\hat{v}_t$$更新自变量$$x_t$$。与AdaGrad、RMSProp以及AdaDelta一样，Adam在优化目标函数时，自变量中的每个分量分别拥有自己的学习率，具体的迭代公式为：

$$
x_t=x_{t-1}-\frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t
$$

在pyTorch中，torch.optim.Adam已实现Adam，**相关的超参数有学习率、beta1和beta2。**
