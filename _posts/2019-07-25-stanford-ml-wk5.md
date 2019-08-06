---
layout: post
title:  "斯坦福机器学习课程笔记（Week 5）"
date:   2019-07-25 11:14:00 -0700
tags:   study-cs machien-learning
---

## 本系列的其它文章

- [斯坦福机器学习课程笔记（Week 1）]({% post_url 2019-06-25-stanford-ml-wk1 %})
- [斯坦福机器学习课程笔记（Week 2）]({% post_url 2019-07-05-stanford-ml-wk2 %})
- [斯坦福机器学习课程笔记（Week 3）]({% post_url 2019-07-13-stanford-ml-wk3 %})
- [斯坦福机器学习课程笔记（Week 4）]({% post_url 2019-07-15-stanford-ml-wk4 %})
- **斯坦福机器学习课程笔记（Week 5）**
- [斯坦福机器学习课程笔记（Week 6）]({% post_url 2019-08-01-stanford-ml-wk6 %})
- [斯坦福机器学习课程笔记（Week 7）]({% post_url 2019-08-05-stanford-ml-wk7 %})
- *斯坦福机器学习课程笔记（Week 8）（敬请期待）*
- *斯坦福机器学习课程笔记（Week 9）（敬请期待）*
- *斯坦福机器学习课程笔记（Week 10）（敬请期待）*
- *斯坦福机器学习课程笔记（Week 11）（敬请期待）*

## 神经网络（续）

### 变量名称和字母

在学习神经网络的代价函数之前，我们先确定一些变量的名称和所用字母

- $$L$$代表网络的总层数
- $$s_l$$代表在第$$l$$层中的神经元个数（不包括bias unit）
- 若神经网络被应用在双类别分类问题，$$y\in\{0, 1\}$$
  - $$h_\Theta(x)\in\mathbb{R}$$
- 若神经网络被应用在多类别分类问题中（假设有$$K$$类，$$K\ge3$$），$$y\in\mathbb{R}^K$$
  - $$h_\Theta(x)\in\mathbb{R}^K$$，用$$(h_\Theta(x))_i$$代表第$$i$$个输出神经元的结果

### 代价函数

Logistic回归中的代价函数如下（使用了regularization）

$$
J(\theta)=-\frac1m[\sum_{i=1}^my^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]+\frac\lambda{2m}\sum_{j=1}^n\theta^2_j
$$

神经网络的代价函数实际上可以看成是Logistic回归中代价函数的推广

$$
J(\Theta)=-\frac1m[\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)}\log(h_\Theta(x^{(i)}))_k+(1-y_k^{(i)})\log(1-(h_\Theta(x^{(i)}))_k)] + \frac\lambda{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_l+1}(\Theta^{(l)}_{ji})^2
$$

### 反向传播（Backpropagation Algorithm）与误差值

为了最小化代价函数，我们需要能够计算

- $$J(\Theta)$$
- $$\frac\partial{\partial\Theta_{ij}^{(l)}}J(\Theta)$$

我们可以通过类似前向传播的方式（通过第$$l$$层的激活值$$a^{(l)}$$来得到第$$l+1$$层的激活值），进行反向传播。我们可以令$$\delta^{(l)}_j$$等于第$$l$$层中第$$j$$个节点的误差值。假设有一个四层的神经网络，那么第四层（输出层）的误差值可以用如下公式表示

$$
\delta^{(4)}_j=a^{(4)}_j-y_j=(h_\Theta(x))_j-y_j
$$

第三层的误差值如下

$$
\delta^{(3)}=(\Theta^{(3)})^T\delta^{(4)}\circ g'(z^{(3)})
$$

其中的“$$\circ$$”符号代表求两个矩阵的逐项积（element-wise multiplication，也叫Hadamard product），$$(A\circ B)_{ij}=A_{ij}B_{ij}$$。

此外，$$g'$$是sigmoid function $$g$$的导数，$$g'(z^{(3)})=g(z^{(3)})(1-g(z^{(3)}))=a^{(3)}\circ(1-a^{(3)})$$。

类似地，第二层的误差值如下

$$
\delta^{(2)}=(\Theta^{(2)})^T\delta^{(3)}\circ g'(z^{(2)})
$$

由于因为第一层是输入层，不存在误差。

如果不使用regularization，可以证明

$$
\frac\partial{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_j^{(l)}\delta_i^{(l+1)}
$$

### 反向传播算法

训练数据可以表示为$$\{(x^{(1)},y^{(1)}),\cdots,(x^{(m)},y^{(m)})\}$$。

设$$\Delta_{ij}^{(l)}=0\quad\text{(for all }l,i,j\text{)}$$，这里的$$\Delta$$是$$\delta$$的大写字母。

重复以下步骤$$m$$次（一次只处理一条训练数据）

- 令$$a^{(1)}=x^{(i)}$$

- 用前向传播法来计算$$l=2,3,\cdots,L$$时的所有$$a^{(l)}$$

- 用训练数据中的$$y^{(i)}$$来计算$$\delta^{(L)}=a^{(L)}-y^{(i)}$$

- 依次计算$$\delta^{(L-1)},\delta^{(L-2)},\cdots,\delta^{(2)}$$（第一层是输入层，不存在误差）

- 最后令$$\Delta_{ij}^{(l)}:=\Delta_{ij}^{(l)}+a_j^{(l)}\delta_i^{(l+1)}$$

  - 向量化如下

    $$\Delta^{(l)}:=\Delta^{(l)}+\delta^{(l+1)}(a^{(l)})^T$$

最后我们令$$D$$等于如下等式

$$
D_{ij}^{(l)}:=
\begin{cases}
\frac1m\Delta_{ij}^{(l)}+\lambda\Theta_{ij}^{(l)}& \text{(if }j\ne0\text{)}\\
\frac1m\Delta_{ij}^{(l)}& \text{(if }j=0\text{)}
\end{cases}
$$

可以证明

$$
\frac\partial{\partial\Theta_{ij}^{(l)}}J(\Theta)=D_{ij}^{(l)}
$$

### 如何理解反向传播法

先回顾一下前向传播法

![image-20190724004945262](/assets/2019-07-25-stanford-ml-wk5/image-20190724004945262.png)

实际上，$$\delta_j^{(l)}$$代表了$$a_j^{(l)}$$的误差大小。可以证明（通过链式法则）

$$
\delta_j^{(l)}=\frac\partial{\partial z_j^{(l)}}\text{cost}(i) \\
\text{cost}(i)=y^{(i)}\log h_\Theta(x^{(i)})+(1-y^{(i)})\log(1-h_\Theta(x^{(i)}))
$$

个人的一点理解，神经网络实际上是将非线性分类中使用高次多项式进行拟合的方法，换成了使用多个函数的复合进行拟合。感觉$$h_\Theta$$实际上可以看成类似$$h^{(1)}_{\Theta^{(1)}}\circ h^{(2)}_{\Theta^{(2)}}\circ\cdots\circ h^{(L-1)}_{\Theta^{(L-1)}}$$这样的一个复合函数，前向传播的过程就是按部就班地求解这个复合函数。至于反向传播，这个是反复运用链式法则对代价函数进行偏微分，因为为了使用梯度下降法，我们首先得知道代价函数的梯度。

![preview](/assets/2019-07-25-stanford-ml-wk5/v2-8e30a45198d332dae959c57e04fdc267_r.jpg)

用上上图中的例子，假设现在我只关心$$a_2^{(3)}$$这个神经元是怎么影响到我们的输出（$$a_1^{(4)}$$），即我想知道$$J(\Theta)$$关于$$\Theta_{12}^{(3)}$$的偏导数，通过链式法则，我们可以列出下列式子

$$
\begin{eqnarray*}

\frac{\partial J(\Theta)}{\partial\Theta_{12}^{(3)}}&=&\frac{\partial J(\Theta)}{\partial a_1^{(4)}}\cdot\frac{\partial a_1^{(4)}}{\partial z_1^{(4)}}\cdot\frac{\partial z_1^{(4)}}{\partial\Theta_{12}^{(3)}} \\
&=&\delta_1^{(4)}\cdot\frac{\partial z_1^{(4)}}{\partial\Theta_{12}^{(3)}} \\

\end{eqnarray*}
$$

根据定义$$z^{(l)}=\Theta^{(l-1)}a^{(l-1)}$$易得

$$
\frac{\partial z^{(4)}}{\partial\Theta^{(3)}}=a^{(3)}
$$

即

$$
\frac{\partial z_1^{(4)}}{\partial\Theta_{12}^{(3)}}=a_2^{(3)}
$$

因此

$$
\frac{\partial J(\Theta)}{\partial\Theta_{12}^{(3)}}
=\delta_1^{(4)}a_2^{(3)}
$$

最后进行“下降”步骤即可（$$\alpha$$为学习率）

$$
\Theta_{12}^{(3)}:=\Theta_{12}^{(3)}-\alpha\cdot\delta_1^{(4)}a_2^{(3)}
$$

[参考资料：《一文弄懂神经网络中的反向传播法》](https://www.cnblogs.com/charlotte77/p/5629865.html)

### 梯度检查（Gradient Checking）

在实现FP和BP算法有时候会存在一些难以发现的bug，因此我们经常需要对我们的梯度进行检查。方法如下

- 将$$\Theta$$展开，设得到的向量为$$\theta\in\mathbb{R}^n$$

  - 此处的$$n$$为向量$$\theta$$的长度

- 计算近似梯度（一般$$\epsilon=10^{-4}$$）
  
  $$
  \begin{eqnarray*}
  &\frac\partial{\partial\theta_n}&J(\theta)\approx\frac{J(\theta_1+\epsilon,\theta_2,\theta_3,\cdots,\theta_n)-J(\theta_1-\epsilon,\theta_2,\theta_3,\cdots,\theta_n)}{2\epsilon} \\
  &\frac\partial{\partial\theta_n}&J(\theta)\approx\frac{J(\theta_1,\theta_2+\epsilon,\theta_3,\cdots,\theta_n)-J(\theta_1,\theta_2-\epsilon,\theta_3,\cdots,\theta_n)}{2\epsilon} \\
  &\vdots& \\
  &\frac\partial{\partial\theta_n}&J(\theta)\approx\frac{J(\theta_1,\theta_2,\theta_3,\cdots,\theta_n+\epsilon)-J(\theta_1,\theta_2,\theta_3,\cdots,\theta_n-\epsilon)}{2\epsilon}
  \end{eqnarray*}
  $$

代码如下

```matlab
for i = 1:n,
    thetaPlus = theta;
    thetaPlus(i) = thetaPlus(i) + EPSILON;
    thetaMinus = theta;
    thetaMinus(i) = thetaMinus(i) - EPSILON;
    gradApprox(i) = (J(thetaPlus) - J(thetaMinus)) / (2 * EPSILON);
end;
```

我们得到近似梯度后，只需要将它与BP算法中产生的梯度（上面“反向传播算法”这部分中的大写字母$$D$$的展开形式）进行对比即可。如果它们俩非常接近，我们就认为我们的BP算法的实现没有问题。当我们检测完我们的BP算法的准确性后，务必**禁用手动的近似梯度检查**，因为手动近似梯度检查会严重拖慢代码的运行效率。

### 随机初始化（Random Initialization）

使用梯度下降法或者其它高级优化算法前，我们都需要将参数初始化。尽管在logistic回归中我们可以将参数初始化为0，但是在神经网络中，初始化参数为0会使神经网络无法工作。

![image-20190724162902770](/assets/2019-07-25-stanford-ml-wk5/image-20190724162902770.png)

简单来讲，神经网络无法工作的原因是，在FP时，全是0的参数会导致隐藏层（$$a_1^{(2)}$$和$$a_2^{(2)}$$）的激活程度相同。又因为隐藏层和参数相同，在BP时会导致每个隐藏层节点的误差（$$\delta$$）相同，于是在更新上图中蓝、红、绿线所示的参数时，大家都按照相同的量进行更新，最后导致所有的参数都变为一个相同的非0值，这在第二次FP时，又会导致隐藏层的激活程度相同，如此往复，神经网络就无法输出令我们满意的结果。

为了解决零初始化产生的问题，我们可以使用随机初始化。简单来讲，就是令

$$
-\epsilon\le\Theta_{ij}^{(l)}\le\epsilon
$$

注意此处的$$\epsilon$$跟梯度检查那部分的无关。下面的公式是得到$$\epsilon$$值的其中一种方法

$$
\epsilon_\text{init}=\frac{\sqrt{6}}{\sqrt{L_\text{in}+L_\text{out}}}
$$

其中$$L_\text{in}=s_l$$、$$L_\text{out}=s_{l+1}$$，分别是$$\Theta^{(l)}$$左右两侧的神经元的数量。
