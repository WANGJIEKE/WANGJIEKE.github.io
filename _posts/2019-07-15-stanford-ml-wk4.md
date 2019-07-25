---
layout: post
title:  "斯坦福机器学习课程笔记（Week 4）"
date:   2019-07-15 22:14:00 -0700
tags:   study-cs machien-learning
---

## 本系列的其它文章

- [斯坦福机器学习课程笔记（Week 1）]({% post_url 2019-06-25-stanford-ml-wk1 %})
- [斯坦福机器学习课程笔记（Week 2）]({% post_url 2019-07-05-stanford-ml-wk2 %})
- [斯坦福机器学习课程笔记（Week 3）]({% post_url 2019-07-13-stanford-ml-wk3 %})
- **斯坦福机器学习课程笔记（Week 4）**
- [斯坦福机器学习课程笔记（Week 5）]({% post_url 2019-07-25-stanford-ml-wk5 %})

## 神经网络（Neural Network）

### 非线性分类问题

![image-20190715115611717](/assets/2019-07-15-stanford-ml-wk4/image-20190715115611717.png)

对于上面的这种比较复杂的非线性分类问题，我们通常需要把feature（特征）给弄成多项式（从已有的feature中组成高次的feature），再使用logistic回归来解决问题。

在这种只有$$x_1$$和$$x_2$$两个feature的情况下还好，如果我们要解决一些应用题，比如说预测房价，这个问题里可能有100个不同的feature。而这100个feature可以组成有5000项的二次多项式，这样的计算是不小的开支。一般来讲，如果有$$n$$个feature，那么对应的二次多项式的项数是以$$\mathcal{O}(n^2)$$（具体来说是$$\frac{n^2}2$$）的速度增长的。

### 模型表示

跟人的大脑类似，我们的神经网络也是由许多“神经元”组成的。

![image-20190715123914237](/assets/2019-07-15-stanford-ml-wk4/image-20190715123914237.png)

在神经网络中，有时我们将$$x_0$$称为bias unit（偏移量），它的值总是等于1。此外，有些人称参数$$\theta$$为“weights”（权重）。此外，我们称$$g(z)$$为激活函数，它是一个非线性函数。上图中的激活函数是sigmoid function，实际上激活函数还有很多其它的选择。

而完整的神经网络则是上面的神经元的组合。

![image-20190715124959823](/assets/2019-07-15-stanford-ml-wk4/image-20190715124959823.png)

注意，中间的hidden layer（隐藏层）可以有多层。我们用$$a_i^{(j)}$$来表示在第$$j$$层的第$$i$$个神经元的激活情况，这里的“激活情况”是指这个神经元产生的输出值。此外，我们用$$\Theta^{(j)}$$来表示针对于从第$$j$$层到第$$j+1$$层的转换的权值矩阵。对于上图中的例子，我们可以列出如下式子

$$
\begin{eqnarray*}
a_1^{(2)}&=&g(\Theta_{10}^{(1)}x_0+\Theta_{11}^{(1)}x_1+\Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3) \\
a_2^{(2)}&=&g(\Theta_{20}^{(1)}x_0+\Theta_{21}^{(1)}x_1+\Theta_{22}^{(1)}x_2+\Theta_{23}^{(1)}x_3) \\
a_3^{(2)}&=&g(\Theta_{30}^{(1)}x_0+\Theta_{31}^{(1)}x_1+\Theta_{32}^{(1)}x_2+\Theta_{33}^{(1)}x_3) \\
h_\Theta(x)&=&a_1^{(3)}=g(\Theta_{10}^{(2)}a_0^{(2)}+\Theta_{11}^{(2)}a_1^{(2)}+\Theta_{12}^{(2)}a_2^{(2)}+\Theta_{13}^{(2)}a_3^{(2)})
\end{eqnarray*}
$$

如果一个神经网络的第$$j$$层中有$$s_j$$个unit（神经元），第$$j+1$$层中有$$s_{j+1}$$个units，那么$$\Theta^{(j)}$$会是一个$$s_{j+1}\times(s_j+1)$$的矩阵。

### 前向传播（Forward Propagation）

对于前面的例子，我们可以定义

$$
z_1^{(2)}=\Theta_{10}^{(1)}x_0+\Theta_{11}^{(1)}x_1+\Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3
$$

如果要向量化上面的例子，我们可以定义向量$$\mathbf{x}$$和向量$$\mathbf{z}^{(2)}$$为

$$
\mathbf{x}=\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix} \qquad
\mathbf{z}^{(2)}=\begin{bmatrix}z_1^{(2)} \\ z_2^{(2)} \\ z_2^{(2)} \end{bmatrix}
$$

根据矩阵乘法的规则，我们可以得到

$$
\mathbf{z}^{(2)}=\Theta^{(1)}\mathbf{x} \in \mathbb{R}^3
$$

为了整齐，我们也可以令$$\mathbf{a}^{(1)}=\mathbf{x}$$，则我们有

$$
\mathbf{z}^{(2)}=\Theta^{(1)}\mathbf{a}^{(1)}
$$

因此

$$
\mathbf{a}^{(2)}=g(\mathbf{z}^{(2)})
$$

往向量$$\mathbf{a}^{(2)}$$中也添加bias unit，即令$$a_0^{(2)}=1$$，使得$$\mathbf{a}^{(2)}\in\mathbb{R}^4$$，然后类似地

$$
\mathbf{z}^{(3)}=\Theta^{(2)}\mathbf{a}^{(2)} \\
h_\Theta(\mathbf{x})=\mathbf{a}^{(3)}=g(\mathbf{z}^{(3)})
$$

以上就是前向传播的一个例子。

前向传播这个名称就是这么得来的，我们从输入层开始，一步步计算“激励值”，传递到隐藏层，最后传递到输出层，就好像将“神经冲动”一层层向目标输出层传递。

此外，神经网络很好地解决了在非线性分类问题中特征膨胀的问题。它不采用高次项来拟合训练集，而是通过隐藏层中的新“特征”来解决非线性分类问题。

![image-20190715145004117](/assets/2019-07-15-stanford-ml-wk4/image-20190715145004117.png)

如果我们只看最后一层的话，我们会发现神经网络在这一层跟之前的logistic回归非常相似，都是用特征作为输入，输出概率。只不过在神经网络中，最后一层的特征，是来自于前面的隐藏层的激励值，而不是输入层的值。简单来讲，神经网络通过隐藏层，从原特征中通过某些方式“抽象”、“总结”出了一些新的，复杂的特征，然后用新的特征进行预测。

![img](/assets/2019-07-15-stanford-ml-wk4/rag_zbGqEeaSmhJaoV5QvA_52c04a987dcb692da8979a2198f3d8d7_Screenshot-2016-11-23-10.28.41.png)

### 神经网络与多类别分类

前面说过，神经网络的输出层跟logistic回归非常类似。所以，对于多类别分类问题，我们的输出层就会有多个神经元，每一个对应一种类别。

![image-20190715161638115](/assets/2019-07-15-stanford-ml-wk4/image-20190715161638115.png)

留意上图中$$h_\Theta(\mathbf{x})$$是一个$$\mathbb{R}^4$$的向量。类似地，为了配合神经网络，我们不再令$$y^{(i)}\in\{1,2,3,4\}$$，而是

$$
\mathbf{y}^{(i)}\in\{\begin{bmatrix}1\\0\\0\\0\end{bmatrix},\begin{bmatrix}0\\1\\0\\0\end{bmatrix},\begin{bmatrix}0\\0\\1\\0\end{bmatrix},\begin{bmatrix}0\\0\\0\\1\end{bmatrix}\}
$$