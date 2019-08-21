---
layout: post
title:  "斯坦福机器学习课程笔记（Week 7）"
date:   2019-08-05 12:32:00 -0700
tags:   study-cs machien-learning
---

## 本系列的其它文章

- [斯坦福机器学习课程笔记（Week 1）]({% post_url 2019-06-25-stanford-ml-wk1 %})
- [斯坦福机器学习课程笔记（Week 2）]({% post_url 2019-07-05-stanford-ml-wk2 %})
- [斯坦福机器学习课程笔记（Week 3）]({% post_url 2019-07-13-stanford-ml-wk3 %})
- [斯坦福机器学习课程笔记（Week 4）]({% post_url 2019-07-15-stanford-ml-wk4 %})
- [斯坦福机器学习课程笔记（Week 5）]({% post_url 2019-07-25-stanford-ml-wk5 %})
- [斯坦福机器学习课程笔记（Week 6）]({% post_url 2019-08-01-stanford-ml-wk6 %})
- **斯坦福机器学习课程笔记（Week 7）**
- [斯坦福机器学习课程笔记（Week 8）]({% post_url 2019-08-12-stanford-ml-wk8 %})
- [斯坦福机器学习课程笔记（Week 9）]({% post_url 2019-08-20-stanford-ml-wk9 %})
- *斯坦福机器学习课程笔记（Week 10）（敬请期待）*
- *斯坦福机器学习课程笔记（Week 11）（敬请期待）*

## 支持向量机（Support Vector Machine）

### 从Logistic回归到支持向量机

对于传统的Logistic回归，我们的预测函数如下

$$
h_\theta(x)=\frac1{1+e^{-\theta^Tx}}
$$

对应的代价函数如下

$$
J^{(i)}(\theta)=-y^{(i)}\log\frac1{1+e^{-\theta^Tx^{(i)}}}-(1-y^{(i)})\log(1-\frac1{1+e^{-\theta^Tx^{(i)}}})
$$

对于支持向量机的代价函数，我们的代价函数是Logistic回归的代价函数的修改版

$$
J(\theta)=\frac1m\sum_{i=1}^m(-y^{(i)}\text{cost}_1(\theta^Tx^{(i)})-(1-y^{(i)})\text{cost}_0(\theta^Tx^{(i)}))+\frac\lambda{2m}\sum_{j=1}^n\theta_j^2
$$

其中$$\text{cost}_1$$和$$\text{cost}_0$$函数是两个分段函数，作用与sigmoid function类似，图像如下

![image-20190804200258632](/assets/2019-08-05-stanford-ml-wk7/image-20190804200258632.png)

我们在支持向量机中一般使用其他方式来表示这个代价函数。我们首先移走$$\frac1m$$和$$\frac\lambda{2m}$$中的$$m$$（因为这两个都是常数，求导时导数为0，对我们的结果没有影响）。其次，我们令

$$
\begin{eqnarray*}
A &=& \sum_{i=1}^m(-y^{(i)}\text{cost}_1(\theta^Tx^{(i)})-(1-y^{(i)})\text{cost}_0(\theta^Tx^{(i)})) \\
B &=& \frac12\sum_{j=1}^n\theta_j^2
\end{eqnarray*}
$$

最后，我们不使用$$A+\lambda B$$来进行正规化，而是使用$$CA+B$$来调节$$A$$与$$B$$之间的权重（我们可以将$$C$$近似看作是$$\frac1\lambda$$），即

$$
J(\theta)=CA+B
$$

最后，我们支持向量机的预测函数为

$$
h_\theta(x)=\begin{cases}
1\qquad\text{if }\theta^Tx\ge0\\
0\qquad\text{otherwise}
\end{cases}
$$

### 大间距分类

有的人也称支持向量机为大间距分类器。不过在介绍大间距分类这个概念之前，我们先来回顾一下支持向量机的代价函数设置

$$
J(\theta)=C\sum_{i=1}^m[y^{(i)}\text{cost}_1(\theta^Tx^{(i)})+(1-y^{(i)})\text{cost}_0(\theta^Tx^{(i)})]+\frac12\sum_{i=1}^n\theta^2_j
$$

当$$y=1$$时，我们希望对应的$$\theta^Tx\ge1$$，而不只是大于0；类似地，我们希望当$$y=0$$时，对应的$$\theta^Tx\le1$$，而不止是小于0。我们的支持向量机希望能有更加明显一点的边距。此外，我们会将$$C$$设置为一个较大的数。

如果我们将$$C$$设置得非常大，我们在最小化代价函数时，会倾向于将

$$
\sum_{i=1}^m[y^{(i)}\text{cost}_1(\theta^Tx^{(i)})+(1-y^{(i)})\text{cost}_0(\theta^Tx^{(i)})]
$$

这一项弄成接近0的数。

在经过这样的设计之后，支持向量机会使用与两侧数据点之间的距离最大的线作为边界，如下图中的黑线所示

![image-20190804202352494](/assets/2019-08-05-stanford-ml-wk7/image-20190804202352494.png)

此外，当我们的$$C$$的数值比较大的时候，我们的支持向量机对异常值会非常敏感，如下图所示，当左下角加入了一个异常值后，支持向量机的决策边界由黑色的线变成了紫色的线。

![image-20190804203057965](/assets/2019-08-05-stanford-ml-wk7/image-20190804203057965.png)

### 大间距分类的数学原理

首先我们来回顾一下向量的内积（点积）。假设有向量$$u=\begin{bmatrix}u_1\\u_2\end{bmatrix}$$，$$v=\begin{bmatrix}v_1\\v_2\end{bmatrix}$$，我们规定$$u$$和$$v$$的内积表示为$$u^Tv$$（或$$u\cdot v$$）。

在介绍如何计算向量内积之前，先了解如何表示一个向量的范数。范数是指满足某些特性的函数的集合（一般这种函数能表示“长度”或者“距离”的含义），在这里我们使用的是向量的欧几里得范数，也称$$L^2$$范数，或者模长，实际上就是指向量的长度。本文中后面的范数如果没有说明，都是指欧几里得范数。向量$$u$$的范数一般写作$$\vert\vert u\vert\vert$$（注意这里是每一侧两条竖线，总共四条线，不要跟绝对值符号弄混，不过实际上绝对值是范数的一种特殊情况）。向量$$u\in\mathbb{R}^n$$的范数的计算公式如下

$$
||u||=\sqrt{\sum_{i=1}^nu_i^2}\in\mathbb{R}
$$

设$$p\in\mathbb{R}$$是向量$$v\in\mathbb{R}^n$$在向量$$u\in\mathbb{R}^n$$上的投影的长度，向量$$u$$和$$v$$的内积的定义如下

$$
u\cdot v=v\cdot u=u^Tv=v^Tu=p\cdot||u||=\sum_{i=1}^nu_iv_i
$$

当$$p>0$$时，代表$$u$$和$$v$$同向；当$$p<0$$时，代表他们反向；当$$p=0$$时，代表他们垂直。

为了简便起见，设我们的支持向量机的参数$$\theta=\begin{bmatrix}0\\\theta_1\\\theta_2\end{bmatrix}\in\mathbb{R}^2$$，我们可以将代价函数中的$$B$$部分展开如下

$$
B=\frac12\sum_{j=1}^2\theta_j^2=\frac12(\sqrt{\sum_{j=1}^2\theta_j^2})^2=\frac12||\theta||^2
$$

对于代价函数中的$$A$$部分，我们重点关心$$\theta^Tx$$这部分。我们可以把它跟向量内积对应起来

$$
\theta^Tx^{(i)}=p^{(i)}\cdot||\theta||=\theta_1x_1^{(i)}+\theta_2x_2^{(i)}
$$

因此，当$$y^{(i)}=1$$时，我们希望$$p^{(i)}\cdot\vert\vert\theta\vert\vert\ge1$$；当$$y^{(i)}=0$$时，希望$$p^{(i)}\cdot\vert\vert\theta\vert\vert\le-1$$。

假设我们用绿线对这些数据进行分类（我们能根据绿线得到对应的$$\theta$$）

![image-20190804221307293](/assets/2019-08-05-stanford-ml-wk7/image-20190804221307293.png)

对于数据点$$x^{(1)}$$和$$x^{(2)}$$，我们能计算出对应的$$p^{(1)}$$和$$p^{(2)}$$。因为当$$y^{(i)}=1$$时，我们希望$$p^{(i)}\cdot\vert\vert\theta\vert\vert\ge1$$；当$$y^{(i)}=0$$时，希望$$p^{(i)}\cdot\vert\vert\theta\vert\vert\le-1$$。如果$$x^{(1)}$$和$$x^{(2)}$$离绿线特别近，那么意味着对应的$$p$$会特别小，那么$$\vert\vert\theta\vert\vert$$要特别大才能使最终的乘积满足要求。但是我们同时希望能够最小化$$\frac12\vert\vert\theta\vert\vert^2$$，也就是说希望$$\vert\vert\theta\vert\vert$$尽可能小，与之前对$$\vert\vert\theta\vert\vert$$的期望有冲突，这说明了这条绿线不是特别好的决策边界。

假设我们修改了参数$$\theta$$，得到了新的决策边界

![image-20190804221836867](/assets/2019-08-05-stanford-ml-wk7/image-20190804221836867.png)

当我们的绿线离数据点较远（间距较大）时，$$p$$较大，因此$$\vert\vert\theta\vert\vert$$不需要特别大也可以让乘积满足要求。

下面是我整理的一些关于线性支持向量机的参考资料

- [支持向量机（Support Vector Machine，SVM）——线性SVM](https://www.cnblogs.com/wuliytTaotao/p/10175888.html)
- [支持向量机（SVM）里的支持向量是什么意思](http://sofasofa.io/forum_main_post.php?postid=1000255)
- [支持向量机通俗导论（理解SVM的三层境界）](https://blog.csdn.net/v_july_v/article/details/7624837)

### 核函数（Kernels）简介

在之前的学习中，我们了解了支持向量机。但是目前我们的支持向量机只能用于线性可分的问题中，对于非线性可分的问题我们是无能为力的。对于非线性可分问题，我们在介绍logistic回归的时候提到过了使用多项式特征，如下图所示

![image-20190805012048238](/assets/2019-08-05-stanford-ml-wk7/image-20190805012048238.png)

我们可以使用高次项来引入更多的特征，但是我们希望有一种更好的解决方法（神经网络是其中一种），对于支持向量机来说，我们有其它的方法来增加我们的特征。

我们可以在样本中随意选择几个点，作为我们的地标（landmark），然后我们通过计算一个特定样本$$x$$与地标的靠近程度来得到不同的特征，如下图所示

![image-20190805013304487](/assets/2019-08-05-stanford-ml-wk7/image-20190805013304487.png)

在这个例子中，我们选择了三个不重合的点作为地标，用字母表示为$$l^{(1)}$$，$$l^{(2)}$$，和$$l^{(3)}$$。然后，我们定义新的特征$$f_i$$如下

$$
f_i=\text{similarity}(x,l^{(i)})=\exp(-\frac{||x-l^{(i)}||^2}{2\sigma^2})=\exp(-\frac{\sum_{j=1}^n(x_j-l_j^{(i)})^2}{2\sigma^2})=K(x,l^{(i)})
$$

其中$$\exp(n)=e^n$$。这个$$\text{similarity}$$函数我们称之为核函数（kernel），可以用字母$$K$$表示。在这个例子中我们使用的是高斯核函数（Gaussian kernel）。

若我们如此设置核函数，当$$x\approx l^{(i)}$$时，$$f_i\approx1$$；当$$x$$和$$l^{(i)}$$相隔特别远的时候，$$f_i\approx0$$。下面是一个核函数的例子，我们可以发现，当$$\sigma$$越大时，核函数的图像越平缓；当$$\sigma$$越小时，核函数的图像越陡峭。

![image-20190805014903664](/assets/2019-08-05-stanford-ml-wk7/image-20190805014903664.png)

从下面的例子可以看出，若参数已知，我们就可以根据地标得到新的特征，最后得到红色的决策边界。

![image-20190805015510816](/assets/2019-08-05-stanford-ml-wk7/image-20190805015510816.png)

### 核函数与支持向量机

我们可以将核函数应用在每一个数据点上。假设我们有$$m$$个数据，我们能得到一个特征向量（feature vector，不是线代里的eigen vector）$$f=\begin{bmatrix}f_0\\f_1\\\vdots\\f_m\end{bmatrix}\in\mathbb{R}^{m+1}$$（其中$$f_0=1$$）。对于每一个数据$$x^{(i)}\in\mathbb{R}^{n+1}$$（算上$$x_0$$），我们都能按照下面的关系得到与之对应的$$f^{(i)}$$（为了简便，下面用$$\text{sim}$$代替$$\text{similarity}$$）

$$
f^{(i)}=\begin{bmatrix}
f_0^{(i)}\\f_1^{(i)}\\f_m^{(2)}\\\vdots\\f_m^{(i)}
\end{bmatrix}=\begin{bmatrix}
1\\
\text{sim}(x^{(i)},l^{(1)})\\
\text{sim}(x^{(i)},l^{(2)})\\
\vdots\\
\text{sim}(x^{(i)},l^{(m)})
\end{bmatrix}
$$

得到新的特征向量后，我们可以将预测函数和代价函数中的$$x$$换成$$f$$。注意，此时我们的参数$$\theta$$的维度也会变成$$m+1$$维。

$$
h_\theta(x)=\begin{cases}
1\qquad\text{if }\theta^Tf\ge0\\
0\qquad\text{otherwise}
\end{cases}\\
J(\theta)=C\sum_{i=1}^my^{(i)}\text{cost}_1(\theta^Tf^{(i)})+(1-y^{(i)})\text{cost}_0(\theta^Tf^{(i)})+\frac12\sum_{j=1}^{m}\theta_j^2
$$

### 偏差方差与支持向量机

在我们的代价函数中，$$C$$起到了一个类似$$\frac1\lambda$$的作用。因此，当$$C$$较大时，一般会出现低偏差高方差；反之，当$$C$$较小时，一般会出现高偏差低方差。

此外，在我们的高斯核函数中，还有一个参数$$\sigma^2$$。当$$\sigma^2$$较大时，核函数更平缓，一般会出现高偏差低方差；反之，当$$\sigma^2$$较小时，容易出现低偏差高方差。

### 支持向量机的使用

当我们使用第三方库去求解$$\theta$$时，我们需要提供$$C$$和$$K$$（核函数）。

有时我们称不使用核函数为“使用线性核函数”（当$$n$$很大，但是$$m$$很小，并且不需要很复杂的决策边界的时候可以考虑使用线性核函数）。

另一个可以选择的核函数时高斯核函数。若我们使用高斯核函数，我们需要提供参数$$\sigma^2$$。需要注意的是，在使用高斯核函数前最好进行特征缩放。

此外，并不是所有的$$\text{similarity}(x,l)$$函数都可以作为核函数。只有满足Mercer定理的函数才能作为核函数，保证有关优化函数的正常运行，并使得代价函数收敛。

最后提一些其它的核函数

- 多项式核函数：$$K(x,l)=(x^Tl+c)^d$$，其中$$c$$是某个常数，$$d$$是次数
- String核
- Chi-square核
- Histogram intersection核

### 多类别分类

我们可以同样使用“one-vs-all”方法，训练出多个分类器，挑选出$$(\theta^{(i)})^Tx$$最大的那个$$i$$对应的$$y$$作为结果。

### 支持向量机与Logistic回归

- 当$$n$$很大，$$m$$很小时，考虑使用logistic回归，或者SVM搭配线性核
- 当$$n$$很小，$$m$$的值适中（10到10000），可以考虑SVM搭配高斯核
- 当$$n$$很小，$$m$$很大时，先增加特征，再使用logistic回归或SVM搭配线性核

神经网络也可以用来解决类似问题，但是通常训练速度相对较慢
