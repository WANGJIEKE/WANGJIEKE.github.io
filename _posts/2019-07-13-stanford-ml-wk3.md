---
layout: post
title:  "斯坦福机器学习课程笔记（Week 3）"
date:   2019-07-13 00:23:00 -0700
tags:   study-cs machien-learning
---

## 本系列的其它文章

- [斯坦福机器学习课程笔记（Week 1）]({% post_url 2019-06-25-stanford-ml-wk1 %})
- [斯坦福机器学习课程笔记（Week 2）]({% post_url 2019-07-05-stanford-ml-wk2 %})
- **斯坦福机器学习课程笔记（Week 3）**
- [斯坦福机器学习课程笔记（Week 4）]({% post_url 2019-07-15-stanford-ml-wk4 %})
- [斯坦福机器学习课程笔记（Week 5）]({% post_url 2019-07-25-stanford-ml-wk5 %})
- [斯坦福机器学习课程笔记（Week 6）]({% post_url 2019-08-01-stanford-ml-wk6 %})
- [斯坦福机器学习课程笔记（Week 7）]({% post_url 2019-08-05-stanford-ml-wk7 %})
- [斯坦福机器学习课程笔记（Week 8）]({% post_url 2019-08-12-stanford-ml-wk8 %})
- [斯坦福机器学习课程笔记（Week 9）]({% post_url 2019-08-20-stanford-ml-wk9 %})
- *斯坦福机器学习课程笔记（Week 10）（敬请期待）*
- *斯坦福机器学习课程笔记（Week 11）（敬请期待）*

## 双类别分类问题（Binary Classification）

### 简介

简单来说双类别分类问题就是输出结果为0或1的问题（$$y\in \{0, 1\}$$）。有时称$$y=0$$为negative class，称$$y=1$$为positive class，或分别用符号“$$-$$”和“$$+$$”表示。当然，分类问题也可以拓展到三种或以上类型的情况。

### 使用线性回归在分类问题中的局限性

如果我们使用阈值来将线性回归产生的连续结果分成几个类别的话，那么额外的训练数据会使我们拟合的直线产生偏差。如下图所示（大于等于0.5则认为结果是“Yes”），在最右侧的数据点加入之前，紫色的拟合直线是比较适合用来预测的，但是在最右侧的数据点之后，新的蓝色的拟合直线产生了较大的偏差。

![image-20190711013323484](/assets/2019-07-13-stanford-ml-wk3/image-20190711013323484.png)

此外，即使训练数据中，所有的实测值（$$y$$值）都是0或1，预测函数$$h_\theta(x)$$的值域依然有可能会大于1或者小于0。

## Logistic回归

### 预测函数（Hypothesis）

为了解决之前提到的，线性回归中$$h_\theta(x)$$的值域不在区间$$[0, 1]$$的问题，我们可以对$$h_\theta(x)$$作如下修改

$$
\begin{eqnarray*}
h_\theta(x)&=&g(\theta^Tx) \\
g(z)&=&\frac{1}{1+e^{-z}}
\end{eqnarray*}
$$

其中，$$g(z)$$也被称为sigmoid function或者logistic function，它的图像如下所示

![image-20190711015822738](/assets/2019-07-13-stanford-ml-wk3/image-20190711015822738.png)

将上面的式子组合，得到了logistic回归中的预测函数

$$
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
$$

这个预测函数输出的内容是，当给定某个$$x$$作为输入时，输出$$y=1$$的**概率**。因此，预测函数也可以写成

$$
h_\theta(x)=P(y=1|x;\theta)
$$

意思是，“当$$x$$发生时，$$y=1$$的概率（$$x$$的值由参数$$\theta$$给定）”。举个例子，$$h_\theta(x_0)=0.7$$代表，对于特定的某个输入$$x_0$$来讲，有70%的概率，它对应的$$y_0=1$$。

此外，因为$$y$$的取值范围为0或1，所以

$$
P(y=0|x;\theta)+P(y=1|x;\theta) = 1
$$

### 决策边界（Decision Boundary）

若有

$$
\begin{eqnarray*}
h_\theta(x)&=&g(\theta^Tx) \\
z&=&\theta^Tx \\
g(z)&=&\frac{1}{1+e^{-z}}
\end{eqnarray*}
$$

则当$$z >= 0$$时，$$h_\theta(x)>=0.5$$，也就是说$$y=1$$的概率更大，反之亦然。那么，我们总能找到某些令$$z=0$$（也就是$$h_\theta(x)=0.5$$）的点、线、面等作为我们区分不同输出的分界（阈值），这就是**决策边界**。

![image-20190711023253907](/assets/2019-07-13-stanford-ml-wk3/image-20190711023253907.png)

值得注意的是，决策边界跟我们的数据集无关，**只跟预测函数$$h_\theta$$和它的参数$$\theta_0,\theta_1,\cdots,\theta_n$$有关**。因为决策边界只与我们的预测函数有关，类似前面的多项式回归，我们的决策边界也可以是非线性的，如下图所示

![image-20190711024205764](/assets/2019-07-13-stanford-ml-wk3/image-20190711024205764.png)

### 代价函数（Cost Function）

与线性回归中类似，数据集可以由下列式子表示

$$
\{(x^{(1)},y^{(1)}), (x^{(2)},y^{(2)}), \cdots , (x^{(m)},y^{(m)})\} \\
$$

且

$$
x = \begin{bmatrix}
x_0 \\ x_1 \\ \vdots \\ x_n
\end{bmatrix} \in \mathbb{R}^{n+1} \\
x_0=1,y\in\{0,1\}
$$

但是，因为预测函数

$$
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
$$

不是线性的，所以如果再仿照线性回归中的方法定义代价函数的话（最小二乘法），得到的$$J(\theta)$$就不再是凸函数了。

![image-20190711030244493](/assets/2019-07-13-stanford-ml-wk3/image-20190711030244493.png)

对于函数$$f$$，若它满足下列两个性质，则称它为线性的

$$
\begin{eqnarray*}
f(x_1+x_2)&=&f(x_1)+f(x_2) \\
f(kx)&=&kf(x)
\end{eqnarray*}
$$

对于函数$$f$$，在它的图像上任取两点$$p$$和$$q$$，若满足下列关系

$$
f(\frac{p+q}{2}) \le \frac{f(p)+f(q)}{2}
$$

则称函数$$f$$为凸函数。

**注意，由于翻译的问题，再加上convex这个单词本身在英语中也有歧义，在这篇文章中的凸函数（Convex Function）都是指满足上述定义的函数。**

因为非凸函数无法应用梯度下降法，所以我们需要另外寻找一个适合logistic回归的代价函数。

$$
\begin{equation}
\text{Cost}(h_\theta(x),y)=
\begin{cases}
-\log(h_\theta(x)) &\text{if }y=1 \\
-\log(1-h_\theta(x)) &\text{if }y=0
\end{cases}
\end{equation}
$$

当$$y=1$$时，这个函数的图像为

![1562964868959](/assets/2019-07-13-stanford-ml-wk3/1562964868959.png)

意思是，当$$y=1$$时，若$$h_\theta(x)=1$$，也就是预测完全正确的时候，那么$$\text{Cost}=0$$；当$$y=1$$，但是$$h_\theta(x)\to 0$$时，也就是错得非常离谱的时候，$$\text{Cost}\to \infty$$。这个函数在预测函数预测正确时给出的代价非常小，在预测错误时给出的代价特别大。

当$$y=0$$时，函数图像为

![1562965209659](/assets/2019-07-13-stanford-ml-wk3/1562965209659.png)

含义与$$y=1$$时的类似，这里就不再赘述。

### 简化的代价函数

这是我们从前面得到的代价函数

$$
J(\theta)=\frac1m\sum_{i=1}^m\text{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\
\begin{equation}
\text{Cost}(h_\theta(x),y)=
\begin{cases}
-\log(h_\theta(x)) &\text{if }y=1 \\
-\log(1-h_\theta(x)) &\text{if }y=0
\end{cases}
\end{equation}
$$

但是我们现在的代价函数比较复杂，其中$$\text{Cost}$$函数是一个分段函数。我们可以将它改写成下面的形式

$$
\text{Cost}(h_\theta(x),y)=-y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))
$$

因此，完整的代价函数如下（这个函数可以由最大似然估计导出）

$$
\begin{eqnarray*}
J(\theta)&=&\frac1m\sum_{i=1}^{m}\text{Cost}(h_\theta(x^{(i)}),y^{(i)}) \\
&=&-\frac1m[\sum_{i=1}^my^{(i)}\log h_\theta(x^{(i)})+(1-y)\log(1-h_\theta(x^{(i)}))]
\end{eqnarray*}
$$

向量化表示如下

$$
\begin{eqnarray*}
h&=&g(X\theta) \\
J(\theta)&=&\frac1m\cdot(-y^T\log(h)-(1-y)^T\log(1-h))
\end{eqnarray*}
$$

同样，我们也可以使用梯度下降法来找出$$J(\theta)$$的最小值。回忆梯度下降法的公式

$$
\begin{eqnarray*}
&\text{repeat \{} \\
&\quad& \theta_j := \theta_j-\alpha \frac\partial{\partial\theta_j}J(\theta) \\
&\text{\}}
\end{eqnarray*}
$$

代入$$J(\theta)$$关于$$\theta_j$$的偏导数得

$$
\begin{eqnarray*}
&\text{repeat \{} \\
&\quad& \theta_j := \theta_j-\alpha \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} \\
&\text{\}}
\end{eqnarray*}
$$

向量化的梯度下降公式如下

$$
\theta:=\theta-\frac\alpha{m}X^T(g(X\theta)-\vec{y})
$$

### 进阶优化算法

#### 常见的优化算法

- 梯度下降法（Gradient Descent）
- Conjugate Gradient
- BFGS
- L-BFGS

后三者相对于梯度下降法具有无需手动设置$$\alpha$$且收敛速度快的优点。

为了使用优化算法，我们需要提供计算$$J(\theta)$$和$$\frac\partial{\partial \theta_j}J(\theta)$$的方式。

#### 在Octave/MATLAB中使用优化算法

假设我们有下列代价函数

$$
\theta=\begin{bmatrix}\theta_1 \\ \theta_2 \end{bmatrix} \\
J(\theta)=(\theta_1-5)^2+(\theta_2-5)^2 \\
\frac\partial{\partial\theta_1}J(\theta)=2(\theta_1-5) \\
\frac\partial{\partial\theta_2}J(\theta)=2(\theta_2-5)
$$

对应的Octave/MATLAB代码为

```matlab
function [jVal, gradient] = costFunction(theta)
jVal = (theta(1) - 5) ^ 2 + (theta(2) - 5) ^ 2;  % 定义代价函数
gradient = zeros(2, 1);                          % 定义代价函数的梯度函数
gradient(1) = 2 * (theta(1) - 5);
gradient(2) = 2 * (theta(2) - 5);
```

之后就可以调用Octave/MATLAB中自带的进阶优化算法，例如`fminunc`函数（Function Minimization Unconstrained）。

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
% 原视频中是'100'，但是在MATHLAB R2019a中提示应该使用数字100而不是字符串'100'

initialTheta = zeros(2, 1);
[optTheta, functionVal, exitFlag] = ...
    fminunc(@costFunction, initialTheta, options);
```

其中`optTheta`代表求出的最优$$\theta$$；`functionVal`代表代入最优$$\theta$$时代价函数$$J(\theta)$$的值；`exitFlag`代表`fminunc`函数的退出状态，若为`1`代表函数收敛。此外，`fminunc`需要确保$$\theta\in\mathbb{R}^d$$且$$d\ge2$$。如果$$\theta$$仅为一个实数的话可能无法使用`fminunc`函数。

## 多类别分类（Multiclass Classification）

为了解决多类别分类问题，我们可以将它转换为多个二元分类问题，这种策略称为One-vs-all，如下图所示

![1562974257512](/assets/2019-07-13-stanford-ml-wk3/1562974257512.png)

其中，$$h_\theta^{(i)}(x)$$代表针对第$$i$$个class的预测函数，即

$$
h_\theta^{(i)}(x)=P(y=i|x;\theta)
$$

我们就可以通过logistic回归得到每个$$h_\theta^{(i)}(x)$$的值。最后，我们查看，对于一个数据，哪个class的预测函数返回的概率最大，我们就认为这个数据最有可能属于这个类型。

## 过拟合问题（The Problem of Overfitting）

![1562975621052](/assets/2019-07-13-stanford-ml-wk3/1562975621052.png)

![1562986599114](/assets/2019-07-13-stanford-ml-wk3/1562986599114.png)

**欠拟合**（也称为“high bias”）的意思是我们的模型没有很好地表现出数据的特点。**过拟合**（也成为“high variance”）则是指在有很多feature的情况下，我们的预测函数可能会与训练数据拟合地特别好，但是这样的预测函数无法应用到新的数据中。

### 解决过拟合问题

1. 减少feature的数量
   - 手动选择保留哪一个feature
   - Model Selection Algorithm
2. 正规化（Regularization）
   - 保留所有的feature，但是减少它们的数值的大小
   - Works well when we have a lot of features, each of which contributes a bit to predicting $$y$$

### 正规化（Regularization）与代价函数

如果我们发现我们的模型过拟合了，我们可以适当减少一些feature的“权重”。假设我们想让下面的函数更加接近二次函数

$$
\theta_0+\theta_1x+\theta_2x^2+\theta_3x^3+\theta_4x^4
$$

我们可以通过减少$$\theta_3$$和$$\theta_4$$的值来让上面的函数更接近二次函数。为了达到这个目的，我们可以将$$\theta_3$$和$$\theta_4$$加入到我们的代价函数中

$$
J(\theta)=\frac1{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+1000\theta_3^2+1000\theta_4^2
$$

如果我们找出代价函数的最小值，我们能发现，当$$J(\theta)$$取得最小值的时候，$$\theta_3$$和$$\theta_4$$的值都是一个很接近0的数字，这样子我们就避免了让我们的模型过拟合。

我们可以将上面的代价函数推广到一般形式

$$
J(\theta)=\frac1{2m}[\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^n\theta_j^2]
$$

其中$$\lambda$$是正规化参数（Regularization Parameter），它的值越大，代表我们越不喜欢过拟合发生在我们的模型中。注意这里的$$j$$时从1开始的，我们故意跳过了$$\theta_0$$这一项。

### 正规化与线性回归

我们可以对梯度下降法作如下修改

$$
\begin{eqnarray*}
&\text{repeat \{} \\
&\quad& \theta_0 := \theta_j-\alpha \frac1m \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)} \\
&\quad& \theta_j := \theta_j-\alpha [\frac1m\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} + \frac\lambda m\theta_j] \quad \text{(for all }j=1,\cdots,n\text{)}\\
&\text{\}}
\end{eqnarray*}
$$

梯度下降法也可以写成下面的形式

$$
\begin{eqnarray*}
&\text{repeat \{} \\
&\quad& \theta_0 := \theta_j-\alpha \frac1m \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)} \\
&\quad& \theta_j := \theta_j(1-\alpha\frac\lambda m)-\alpha\frac1m\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}\quad \text{(for all }j=1,\cdots,n\text{)}\\
&\text{\}}
\end{eqnarray*}
$$

同样地，标准方程（Normal Equation）也可以进行正规化

$$
\theta=(X^TX+\lambda
\begin{bmatrix}
0& & &\\
& 1& \\
& & 1& \\
& & & \ddots& \\
& & & & 1&
\end{bmatrix}
)^{-1}X^Ty
$$

新的标准方程中用到的矩阵是$$(n+1) \times (n+1)$$的矩阵。

我们之前知道，若$$m\le n$$（即训练数据的数量小于feature的数量时），矩阵$$X^TX$$不可逆，无法使用标准方程。但是因为我们对其进行了正规化，得到的新矩阵是可逆的，可以放心使用标准方程求解。

### 正规化与Logistic回归

类似地，我们也可以对Logistic回归的代价函数作如下修改

$$
J(\theta)=-[\frac1m\sum_{i=1}^my^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]+\frac\lambda{2m}\sum_{j=1}^n\theta_j^2
$$

我们同时对梯度下降法作如下修改

$$
\begin{eqnarray*}
&\text{repeat \{} \\
&\quad& \theta_0 := \theta_j-\alpha \frac1m \sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)} \\
&\quad& \theta_j := \theta_j-\alpha [\frac1m\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} + \frac\lambda m\theta_j] \quad \text{(for all }j=1,\cdots,n\text{)}\\
&\text{\}}
\end{eqnarray*}
$$

### 正规化与进阶优化算法

（因为我这里没办法弄出吴教授那种代码和数学公式混排的，这里就直接上图了）
![1562995180342](/assets/2019-07-13-stanford-ml-wk3/1562995180342.png)
