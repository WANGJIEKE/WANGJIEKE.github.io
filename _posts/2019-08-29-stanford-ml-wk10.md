---
layout: post
title:  "斯坦福机器学习课程笔记（Week 10）"
date:   2019-08-29 19:53:00 -0700
tags:   study-cs machien-learning
---

## 本系列的其它文章

- [斯坦福机器学习课程笔记（Week 1）]({% post_url 2019-06-25-stanford-ml-wk1 %})
- [斯坦福机器学习课程笔记（Week 2）]({% post_url 2019-07-05-stanford-ml-wk2 %})
- [斯坦福机器学习课程笔记（Week 3）]({% post_url 2019-07-13-stanford-ml-wk3 %})
- [斯坦福机器学习课程笔记（Week 4）]({% post_url 2019-07-15-stanford-ml-wk4 %})
- [斯坦福机器学习课程笔记（Week 5）]({% post_url 2019-07-25-stanford-ml-wk5 %})
- [斯坦福机器学习课程笔记（Week 6）]({% post_url 2019-08-01-stanford-ml-wk6 %})
- [斯坦福机器学习课程笔记（Week 7）]({% post_url 2019-08-05-stanford-ml-wk7 %})
- [斯坦福机器学习课程笔记（Week 8）]({% post_url 2019-08-12-stanford-ml-wk8 %})
- [斯坦福机器学习课程笔记（Week 9）]({% post_url 2019-08-20-stanford-ml-wk9 %})
- **斯坦福机器学习课程笔记（Week 10）**
- [斯坦福机器学习课程笔记（Week 11）]({% post_url 2019-08-29-stanford-ml-wk11 %})

## 处理大数据

### 随机梯度下降法（Stochastic Gradient Descent）

传统的梯度下降法在应用到海量的数据时，存在一些问题。回忆梯度下降公式如下

$$
\theta_j:=\theta_j-\alpha\frac1m\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j
$$

我们在计算每一个梯度时，都要进行求和。当$$m$$特别大时，求和上的开销非常大。这种梯度下降也叫批量梯度下降（Batch Gradient Descent）。此外，我们还有另一种方法，叫做随机梯度下降（Stochastic Gradient Descent），算法如下

$$
\text{cost}\big(\theta,(x^{(i)},y^{(i)})\big)=\frac12\big(h_\theta(x^{(i)})-y^{(i)}\big)^2\\
J_\text{train}(\theta)=\frac1m\sum_{i=1}^m\text{cost}\big(\theta, (x^{(i)},y^{(i)})\big) \\
\begin{eqnarray*}
&\text{1. }&\text{Randomly shuffle dataset} \\
&\text{2. }&\text{Repeat \{} \\
&&\quad\text{for }i=1,\dots,m \text{ \{} \\
&&\qquad\theta_j:=\theta_j-\alpha\big(h_\theta(x^{(i)})-y^{(i)}\big)x_j^{(i)} \\
&&\qquad\quad\text{(for }j=0,\dots,n\text{)}\\
&&\quad\text{\}}\\
&&\text{\}}
\end{eqnarray*}
$$

其中，$$\frac\partial{\partial\theta_j}\text{cost}\big(\theta,(x^{(i)},y^{(i)})\big)=\big(h_\theta(x^{(i)})-y^{(i)}\big)x_j^{(i)}$$。

由算法的定义可以看出，这个随机梯度下降的随机之处是，每次求梯度时，都是只求针对某一个训练数据的梯度。

![image-20190829181153248](/assets/2019-08-29-stanford-ml-wk10/image-20190829181153248.png)

如果我们画出损失函数的图像，我们会发现，随机梯度下降的图像（紫色线）相比批量梯度下降（红色线）更曲折。并且随机梯度下降没那么容易达到最小值，更多时候是在最小值附近的某个范围内，不过这样也足够准确了。

至于随机梯度下降中的外层循环，根据数据集的大小，只需要循环1次（对于大量数据）到10次（对于少量数据）。

### 小批量梯度下降（Mini-Batch Gradient Descent）

批量梯度下降在每次求梯度时使用$$m$$个数据，随机梯度下降每次使用1个数据，而小批量梯度下降每次使用$$b$$个数据（$$1<b<m$$）。小批量梯度下降可以看成是前面两种梯度下降法的折中。

$$
\begin{eqnarray*}
&&\text{Repeat \{} \\
&&\quad\text{for }i=1,1+b,1+2b,\dots,m-b+1 \text{ \{} \\
&&\qquad\theta_j:=\theta_j-\alpha\frac1b\sum_{k=1}^{i+b-1}\big(h_\theta(x^{(k)})-y^{(k)}\big)x_j^{(k)} \\
&&\qquad\quad\text{(for }j=0,\dots,n\text{)}\\
&&\quad\text{\}}\\
&&\text{\}}
\end{eqnarray*}
$$

小批量梯度下降如果采用向量化实现的话，效率甚至可以超过随机梯度下降。但是同样，小批度梯度下降也引入了新的变量$$b$$。

### 随机梯度下降的收敛性

我们知道，在运用批量梯度下降的时候，我们可以通过画出迭代次数和损失函数的图像来查看损失函数是否收敛。对于随机梯度下降，我们可以在每一次迭代的时候，更新$$\theta$$前，计算出$$\text{cost}\big(\theta(x^{(i)},y^{(i)})\big)$$的值并储存。然后在每$$M$$次迭代（比如说$$M=1000$$）时，将前$$M$$次迭代的$$\text{cost}\big(\theta(x^{(i)},y^{(i)})\big)$$函数的平均值绘制出来即可。下面是一些列子

![image-20190829184657569](/assets/2019-08-29-stanford-ml-wk10/image-20190829184657569.png)

靠上的两个情况是正常情况。左下角的图，在每1000次循环绘制一次时，图像比较杂乱。如果我们改用5000次，此时会有两种可能，损失函数在缓慢收敛（红色），或者损失函数没有收敛（紫色）。如果你看到的图像是右下角的，说明损失函数在发散，此时应该降低学习率。

最后，因为我们的学习率$$\alpha$$是恒定的，随机梯度下降很难收敛到最小值，而是在最小值附近浮动。如果我们希望它能收敛到最小值，我们可以让学习率随迭代次数减少。

![image-20190829185123204](/assets/2019-08-29-stanford-ml-wk10/image-20190829185123204.png)

## 在线学习

在线学习可以帮助我们处理“动态”的数据集（即有大量源源不断新数据注入的数据集）。它的定义大致如下

$$
\begin{eqnarray*}
&&\text{Repeat \{} \\
&&\quad\text{Get }(x, y)\text{ corresponding to this user}\\
&&\quad\text{Update }\theta\text{ using }(x, y) \\
&&\qquad \theta_j:=\theta_j-\alpha\big(h_\theta(x)-y\big)x_j\\
&&\text{\}}
\end{eqnarray*}
$$

注意这里我们没有加上标，因为对于在线学习来讲，数据是用完即弃的（因此使用它的前提也是能够获得源源不断的新数据）。

## Map-Reduce and Data Parallelism

Map-Reduce的中心思想就是将整个机器学习的运算分摊到过个机器上来完成。

![image-20190829192635790](/assets/2019-08-29-stanford-ml-wk10/image-20190829192635790.png)

上图的例子中，将原本的400次循环的求和分摊到了4个机器上。实际上，很多机器学习算法都是可以被拆分到多个机器上同时运行的。

![image-20190829193206828](/assets/2019-08-29-stanford-ml-wk10/image-20190829193206828.png)

除了可以将数据分散到多个计算机外，因为现代计算机的CPU常常是多核的，我们也可以将数据分散到每一个核心上面。

![image-20190829193442913](/assets/2019-08-29-stanford-ml-wk10/image-20190829193442913.png)
