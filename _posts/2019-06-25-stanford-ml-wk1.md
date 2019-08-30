---
layout: post
title:  "斯坦福机器学习课程笔记（Week 1）"
date:   2019-06-25 03:45:00 -0700
tags:   study-cs machien-learning
---

本笔记基于Coursera上的[Stanford机器学习公开课](https://www.coursera.org/learn/machine-learning/)，这个免费的课程是Stanford CS 229的简化版。

## 本系列的其它文章

- **斯坦福机器学习课程笔记（Week 1）**
- [斯坦福机器学习课程笔记（Week 2）]({% post_url 2019-07-05-stanford-ml-wk2 %})
- [斯坦福机器学习课程笔记（Week 3）]({% post_url 2019-07-13-stanford-ml-wk3 %})
- [斯坦福机器学习课程笔记（Week 4）]({% post_url 2019-07-15-stanford-ml-wk4 %})
- [斯坦福机器学习课程笔记（Week 5）]({% post_url 2019-07-25-stanford-ml-wk5 %})
- [斯坦福机器学习课程笔记（Week 6）]({% post_url 2019-08-01-stanford-ml-wk6 %})
- [斯坦福机器学习课程笔记（Week 7）]({% post_url 2019-08-05-stanford-ml-wk7 %})
- [斯坦福机器学习课程笔记（Week 8）]({% post_url 2019-08-12-stanford-ml-wk8 %})
- [斯坦福机器学习课程笔记（Week 9）]({% post_url 2019-08-20-stanford-ml-wk9 %})
- [斯坦福机器学习课程笔记（Week 10）]({% post_url 2019-08-29-stanford-ml-wk10 %})
- [斯坦福机器学习课程笔记（Week 11）]({% post_url 2019-08-29-stanford-ml-wk11 %})

## 机器学习的定义

Tom Mitchell对机器学习的定义

> A computer program is said to learn from experience $$E$$ with respect to some class of tasks $$T$$ and performace measure $$P$$, if its performance at tasks in $$T$$, as measured by $$P$$, improves with experience $$E$$.

一般来讲，任何机器学习问题可以被分为两大类，监督（Supervised）学习和非监督（Unsupervised）学习。

## Supervised Learning

- Given a data set and **already know waht our correct output should look like**
- Having the idea that there is a relationship between the input and the output

### Regression Problems

- Trying to predict results within a continuous output

- Map input to some continuous function

- 在数据点中找出一条拟合曲线

  ![image-20190623062503518](/assets/2019-06-25-stanford-ml-wk1/image-20190623062503518.png)

### Classification Problems

- Trying to predict results in a discrete output
- Map input into discrete categories
- 在数据点中找出某种分类规则

  ![image-20190623062542950](/assets/2019-06-25-stanford-ml-wk1/image-20190623062542950.png)

  ![image-20190623062555346](/assets/2019-06-25-stanford-ml-wk1/image-20190623062555346.png)

### 例子

- Regression：根据房子的面积预测它的价格（输出是连续的）
- Classification：根据房子的面积预测它的成交价是高于标价还是低于标价（输出是离散的，只有高于标价和低于标价两种情况）

## Unsupervised Learning

- Allows us to approach problems with little or no idea what our results should look like
- We can derive structure from data where we don't necessarily know the effect of the variables
- No feedback based on the prediction results

### Clustering

- Find a way to group data that are somehow similar or related by different variables

  ![image-20190623071457793](/assets/2019-06-25-stanford-ml-wk1/image-20190623071457793.png)

### Non-Clustering ("Cocktail Party Algorithm")

- Allow you to find structure in a chaotic environment

---

## Linear Regression with One Variable: Model Representation

### Terminology

- Training Set：数据集
- $$m$$：数据集的大小
- $$x$$：输入（"input" variable or feature）
- $$y$$：输出（"output" variable or feature）
- $$(x, y)$$：单个训练数据
  - $$(x^{(i)}, y^{(i)})$$：第$$i$$个训练数据
- $$h$$：Hypothesis Function from Learning Algorithm
  - $$y = h(x)$$（此处的$$y$$是估计值）

### Process

![image-20190624232804677](/assets/2019-06-25-stanford-ml-wk1/image-20190624232804677.png)

### 如何表示$$h$$

- $$h_\theta(x) = \theta_0 + \theta_1x$$，$$h_\theta(x)$$有时会简写为$$h(x)$$
  - $$\theta_0$$，$$\theta_1$$，……，$$\theta_i$$这一系列的值称为模型的参数（parameter）
- 上面的$$h_\theta(x)$$可以用于单变量线性回归模型

![image-20190624233255971](/assets/2019-06-25-stanford-ml-wk1/image-20190624233255971.png)

## Linear Regression with One Variable: Cost Function

### Cost Function介绍

- 选取不同的参数，会改变我们的$$h$$
- 通过选取适当的参数，让我们的$$h_\theta(x^{(i)})$$尽可能接近我们的$$y^{(i)}$$（估计值与实际值的“误差”最小）
- 我们可以将参数和“误差”的关系，用一个新的函数$$J(\theta_0, \theta_1)$$表示，称为Cost Function，也称为Square Error Function
  - {% raw %}$${J(\theta_0, \theta_1)}={{1\over{2m}}{\sum_{i=1}^{m}}{(h_\theta(x^{(i)})-y^{(i)})}^2}$${% endraw %}
    - $$h_\theta(x^{(i)}) = \theta_0 + \theta_1x^{(i)}$$
    - 这个函数只与$$\theta_0$$和$$\theta_1$$有关，跟$$x$$和$$y$$无关
  - 我们的目标是找出函数$$J$$的最小值

### Cost Function的图像

当我们的Cost Function有两个参数时，它的图像是一个三维的平面

![image-20190625010705411](/assets/2019-06-25-stanford-ml-wk1/image-20190625010705411.png)

除了三维图像，我们也可以用Contour Plot来表示我们的Cost Function

![image-20190625010933683](/assets/2019-06-25-stanford-ml-wk1/image-20190625010933683.png)

跟地理中的等高线图类似，在同一条线上的$$J(\theta_0, \theta_1)$$的值相同，在这张图中，蓝色代表$$J(\theta_0, \theta_1)$$的值较小的区域，红色代表较大的区域

也可以将Contour Plot看成是前面的三维图的俯视图

#### 图像的局限性

- 我们更希望有软件/算法能直接找出函数$$J$$的最小值，而不是通过画图的方式来确定
- 当参数个数大于2时，我们无法画出函数的图像

## Linear Regression with One Variable: Gradient Descent

- 存在函数$$J(\theta_0, \theta_1, {\cdots}, \theta_n)$$
- 希望获得$$\theta_0, \theta_1, {\cdots}, \theta_n$$，使得$$J(\theta_0, \theta_1, {\cdots}, \theta_n)$$的值最小

### 梯度下降算法的直观理解

想象你在一座山上，在你每一步的距离是恒定的情况下，想用最短的时间前往最低点，那么办法就是，你每一步，都是要确保你在往最陡峭的那个方向，向下走

![image-20190625014207770](/assets/2019-06-25-stanford-ml-wk1/image-20190625014207770.png)

需要注意的是，在使用梯度下降法的时候，不同的起始点，可能会导致算法产生不同的结果（到达不同的local minimum）；但是对于线性回归模型来讲，我们得到的函数$$J$$是凸函数（convex function），不存在这种多个极小值的情况

![image-20190625015018067](/assets/2019-06-25-stanford-ml-wk1/image-20190625015018067.png)

### 梯度下降法的算法和数学原理

*我们使用$$:=$$符号来表示“赋值”操作（跟编程语言中使用`=`进行赋值操作不同）*

$$ \theta_i:=\theta_i - \alpha{\partial\over{\partial\theta_i}}{J(\theta_0, \theta_1, \cdots, \theta_n)}\hspace{4ex} \text{(looping simultaneously for all } i \text{ until converge)} $$

- $$\alpha$$是learning rate（相当于一步迈多大）

  - 如果$$\alpha$$太小，梯度下降的速度会很慢
  - 如果$$\alpha$$太大，算法可能无法收敛（converge），甚至获得发散（diverge）的结果

- $${\partial \over{\partial \theta_i}}J(\theta_0, \theta_1, {\cdots}, \theta_n)$$代表了函数$$J$$沿$$\theta_i$$方向的偏导数

- 所有参数必须同时进行更新（以两个参数的函数$$J$$为例）

    $$\text{temp0} := \theta_0 - \alpha{\partial\over{\partial\theta_0}}{J(\theta_0, \theta_1)}\\\text{temp1} := \theta_1 - \alpha{\partial\over{\partial\theta_1}}{J(\theta_0, \theta_1)}\\\theta_0 := \text{temp0}\\\theta_1 := \text{temp1}$$

- 即使$$\alpha$$固定，梯度下降依然能收敛到极小值的原因，是因为我们将学习率与导数相乘，当我们靠近极值时，导数的值减少，导致总的步长减少，最终到达极值

### 梯度下降法在线性回归中的应用

- 求出所需的偏导数：$${\partial\over{\partial\theta_0}}{J(\theta_0, \theta_1)}={1\over{m}}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})$$以及$${\partial\over{\partial\theta_1}}{J(\theta_0, \theta_1)}={1\over{m}}\sum_{i=1}^{m}x^{(i)}(h_\theta(x^{(i)})-y^{(i)})$$
- 代入所得的偏导数到算法中
- 迭代

#### Batch Gradient Descent 批量梯度下降

- “Batch” means each step of gradient descent uses all the training examples
- 前面介绍的算法中在每一次迭代时，都遍历了所有数据点，这种就属于批量梯度下降

## 补充内容

- [如何直观形象地理解方向导数与梯度以及它们之间的关系](https://www.zhihu.com/question/36301367/answer/142096153)
- [微积分中，符号$$d$$与符号$$\partial$$的区别是什么？](https://www.zhihu.com/question/22470793/answer/21497265)
