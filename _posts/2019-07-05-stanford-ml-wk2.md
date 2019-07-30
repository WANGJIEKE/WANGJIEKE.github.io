---
layout: post
title:  "斯坦福机器学习课程笔记（Week 2）"
date:   2019-07-05 13:21:00 -0700
tags:   study-cs machien-learning
---

## 本系列的其它文章

- [斯坦福机器学习课程笔记（Week 1）]({% post_url 2019-06-25-stanford-ml-wk1 %})
- **斯坦福机器学习课程笔记（Week 2）**
- [斯坦福机器学习课程笔记（Week 3）]({% post_url 2019-07-13-stanford-ml-wk3 %})
- [斯坦福机器学习课程笔记（Week 4）]({% post_url 2019-07-15-stanford-ml-wk4 %})
- [斯坦福机器学习课程笔记（Week 5）]({% post_url 2019-07-25-stanford-ml-wk5 %})

## 多变量线性回归

### Notation

- $$n$$代表变量的数量
- $$x^{(i)}$$代表第$$i$$个训练数据的输入
- $$x^{(i)}_j$$代表第$$i$$个训练数据的输入中的第$$j$$个变量（feature）

### Hypothesis

$$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n$$

为了方便，我们定义$$x_0=1$$，即$$x^{(i)}_0=1$$

$$x$$和$$\theta$$也可以写成向量的形式

$$ \mathbf{x} =\begin{bmatrix} x_{0} \\ x_{1} \\ \vdots \\ x_{n} \end{bmatrix} \in \mathbb{R}^{n+1}$$，$$ \mathbf{\theta} = \begin{bmatrix} \theta_{0} \\ \theta_{1} \\ \vdots \\ \theta_{n} \end{bmatrix} \in \mathbb{R}^{n+1} $$

根据矩阵乘法， $$h_\theta(x)$$ 也可以写成下面的形式

$$h_\theta(x)=\mathbf{\theta}^T\mathbf{x}$$

### 多变量梯度下降

$$ \theta_j:=\theta_j - \alpha{\partial\over{\partial\theta_j}}{J(\mathbf{\theta})}\hspace{4ex} \text{(looping simultaneously for all } j \text{ until converge)} $$

且

$${\partial\over{\partial{\theta_j}}}J(\mathbf{\theta})={1\over{m}}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$$

（与之前的算法非常类似）

## 梯度下降的实践技巧

### Feature Scaling

中心思想：**确保不同的变量的范围大致相同**

如果不这么做的话，算法需要更多的时间才能收敛。

拿两个变量的情况举例子。假设$$x_1$$代表房子的大小（范围$$[0, 2000]$$），$$x_2$$代表房子中卧室的数量（范围$$[1, 5]$$），那么最后得到的cost function等高线图很有可能是个特别扁的椭圆形，不利于梯度下降算法

![image-20190704052138071](/assets/2019-07-05-stanford-ml-wk2/image-20190704052138071.png)

从教授在slides中的标注可以看出，当我们的图形特别扁的时候，在下降时很容易出现之字形下降的情况。

解决方法也很简单，比如说我们可以把所有变量都缩放到$$[-1, 1]$$的范围内（如果范围太小，比如$$[-0.001, 0.001]$$，适当放大也是可以的）。

### Mean Normalization

中心思想：将每个变量$$x_j$$用$${x_j - \mu_j}\over{s_j}$$替换，其中$$\mu_j$$是$$x_j$$的平均值，$$s_j$$是$$x_j$$的标准差（或最大值和最小值的差）。

经过这个操作之后，该变量的平均值会接近0（无需对$$x_0$$进行这个操作）。

### 如何确认梯度下降法是否在正常运作

因为问题的不同，到达收敛所需的迭代次数也不同。

方法：绘制$$J(\mathbf{\theta})$$与迭代次数的关系图

![image-20190704060048666](/assets/2019-07-05-stanford-ml-wk2/image-20190704060048666.png)

如果梯度下降法在正常工作，那么$$J(\mathbf{\theta})$$应该是随迭代次数的增加而下降的。如果你看到的图形跟下图中的类似，应该考虑适当降低学习率$$\alpha$$。

![image-20190704060455550](/assets/2019-07-05-stanford-ml-wk2/image-20190704060455550.png)

![image-20190704060556623](/assets/2019-07-05-stanford-ml-wk2/image-20190704060556623.png)

大小恰到好处的$$\alpha$$会使得$$J(\mathbf{\theta})$$的值在每一次迭代后都会降低。但是如果$$\alpha$$太小，算法运行的速度也会比较慢。常见的学习率有0.001，0.003，0.01，0.03，0.1，0.3和1等（三倍）。

## 多项式回归

### Feature的选择

有时我们可以通过定义新的feature来获得更优的模型。假设我们手头上有很多地的长和宽的数据，需要构建一个模型以预测它们的价值，与其将长和宽看成两个单独的变量（feature），不如设置一个新的变量，设它为长和宽的积（也就是地的面积），这么做得到的模型可能更加符合实际要求。

### 回归方式的选择

有时，我们的数据不太适合使用线性回归，比如下面的例子

![image-20190704062949135](/assets/2019-07-05-stanford-ml-wk2/image-20190704062949135.png)

对于这个数据集，我们可以考虑用$$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_1^2+\theta_3x_1^3$$或者$$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2\sqrt{x_1}$$进行拟合（不用二次多项式的原因是因为二次多项式不是在定义域内单调递增，不太符合房价随面积增加而增加的常识）。在上面的这两个hypothesis中我们就人为地创建了额外的feature，以帮助我们建立模型。

当我们使用这种方式建立模型的时候，scaling就变得至关重要了，如果$$x_1\in[1, 1000]$$，那么$$x_1^2\in[1, 1000^2]$$，$$x_1^3\in[1, 1000^3]$$，我们不进行缩放的话$$J(\mathbf{\theta})$$对应的等高线图会非常扁，影响算法的效率。

## Normal Equation

通过计算直接找出最优的$$\theta$$。可以通过直接令偏导数等于0解得，即令$${\partial\over{\partial\theta_j}}J(\mathbf{\theta})=0\text{ for all }j$$，求解$$\theta_0, \theta_1, \cdots, \theta_n$$。

假设我们有$$m$$条训练数据，每条数据有$$n$$个feature，则我们可以将每一条训练数据用一个向量表示

$$\mathbf{x}^{(i)}=\begin{bmatrix} x_0^{(i)} \\ x_1^{(i)} \\ x_2^{(i)} \\ \vdots \\ x_n^{(i)} \end{bmatrix} \in \mathbb{R}^{n+1}$$

接下来，我们可以构建一个design matrix，用大写字母$$X$$表示，即

$$X=\begin{bmatrix} (\mathbf{x}^{(1)})^T \\ (\mathbf{x}^{(2)})^T \\ \vdots \\ (\mathbf{x}^{(m)})^T \end{bmatrix} = \begin{bmatrix} x_0^{(1)} &x_1^{(1)} &x_2^{(1)} &\cdots &x_n^{(1)} \\ x_0^{(2)} &x_1^{(2)} &x_2^{(2)} &\cdots &x_n^{(2)} \\ \vdots &\vdots &\vdots &\ddots &\vdots \\ x_0^{(m)} &x_1^{(m)} &x_2^{(m)} &\cdots &x_n^{(m)} \end{bmatrix} $$

注意，$$X$$的每一行都是对应的$$x^{(i)}$$的转置。

我们再将训练集的观测结果用另一个向量$$\mathbf{y}$$表示，即

$$\mathbf{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}$$

那么$$\mathbf{\theta}$$的值可以由以下公式求出（证明见文章末尾）

$$\mathbf{\theta}=(X^TX)^{-1}X^T\mathbf{y}$$

当我们使用normal equation时，不需要进行feature scaling。

### 与梯度下降法的对比

| Gradient Descent                | Normal Equation                                              |
| ------------------------------- | ------------------------------------------------------------ |
| 需要选择$$\alpha$$                | 不需要选择$$\alpha$$                                           |
| 需要多次迭代                    | 不需要迭代                                                   |
| 时间复杂度为$$\mathcal{O}(kn^2)$$ | 需要计算$$(X^TX)^{-1}$$，常见实现的时间复杂度是$$\mathcal{O}(n^3)$$ |
| 当$$n$$特别大时依然可用           | 当$$n$$特别大时速度慢（大于10000）                             |

### Noninvertibility

如果$$X^TX$$不可逆，有以下几种可能

- Redundant feature，矩阵$$X$$中存在线性相关的列
  - 例如第一个feature是额定功率$$P$$，第二个是额定电流$$I$$，假设电压相同$$U$$，那么$$P$$就是$$I$$和$$U$$的线性组合（$$P=UI$$），换句话说，$$W$$和$$I$$是线性相关的
- 变量太多（$$m\le n$$）
  - 删去一些feature，或者使用regularization

## Octave/MATLAB编程

### 基本使用

- `2 ^ 6`用于计算$$2^6$$的结果
- `%`符号用于注释
- `1 ~= 2`用于检测不等关系（不是其它编程语言中常见的`!=`符号）
- `&&`表示逻辑与、`||`表示逻辑或，`xor(1, 0)`表示异或
- 可以通过输入`PS1('>> ')`将提示符改为引号内的内容
- 分号`;`会禁用Octave/MATLAB的实时输出
- 使用`disp(a)`来显示变量`a`中的内容
- 格式化字符串跟C语言中的类似：`disp(sprintf('2 decimals: %0.2f', a))`

### 矩阵和向量

```Matlab
A = [1 2; 3 4; 5 6]
```

可以生成

$$ A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} $$

```matlab
v = 1:0.1:2
```

生成

$$ v = \begin{bmatrix} 1 & 1.1 & 1.2 & \cdots & 1.8 & 1.9 & 2 \end{bmatrix} $$

```matlab
v = 1:6
```

生成

$$ v = \begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 \end{bmatrix} $$

```matlab
ones(2, 3)
```

生成

$$ \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix} $$

类似地

```matlab
zeros(1, 3)
```

生成

$$ \begin{bmatrix} 0 & 0 & 0 \end{bmatrix} $$

可以通过`rand(n, m)`产生$$n\times m$$的随机数（在区间$$[0, 1]$$中均匀分布）矩阵，可以通过`randn(n, m)`产生$$n\times m$$的符合标准正态分布（高斯分布，平均数为零，方差为一）的随机数矩阵

```matlab
w = -6 + sqrt(10)*(randn(1, 10000));
hist(w)
hist(w, 50)  % 指定该直方图有50个bucket
```

上面这段代码可以将随机变量`w`的直方图显示出来

通过`eye(n)`获得对应的单位矩阵$$I_n$$

可以通过`help`来查看帮助，例如`help eye`可以查看关于`eye`的帮助

`size(A)`可以获得矩阵`A`的行数和列数（`size(A, 1)`获得行数，`size(A, 2)`获得列数）

`length(v)`可以获得向量`v`的维度

### 存取数据

`load filename`或者`load('filename')`可以从文件中加载数据，储存在名为`filename`的变量中

`who`可以列出当前作用域中的变量，`whos`可以查看更多关于当前作用域中变量的细节，例如大小和类型

`clear variable_name`可以删去某个变量，若不提供`variable_name`则删除所有变量

假设`x`是一个$$50 \times 1$$的矩阵，`x(1:10)`会返回`x`的前10行

`save filename.mat variable`可以将变量`v`以二进制形式保存到`filename.mat`文件中，若要以文本形式保存，则应在命令的最后加上 `-ascii`参数，即`save filename.mat variable -ascii`

假设变量`A`为矩阵，`A(i, j)`能获得矩阵`A`中第`i`行第`j`列的值

`A(i, :)`代表获得整个第`i`行，`A(:, j)`代表获得整个第`j`列

`A([1 3], :)`代表获得`A`的第1行和第3行

`A(:, 2) = [10; 11; 12]`可以将`A`的第二列替换为`[10; 11; 12]`

`A = [A, [100; 101; 102]]`可以往`A`中增加一列`[100; 101; 102]`

`A(:)`代表将`A`中的所有值存入单个向量中

假设`A`和`B`为合适的矩阵，则`C = [A B]`（或`C = [A, B]`）代表将`B`中的列加入到`A`的右侧，再将获得的新矩阵存到`C`中，类似地，`C = [A; B]`代表将`B`放到`A`的下面

### 计算数据

假设`A`和`B`是两个合适的矩阵，`A .* B`则代表将`A`中的每个值，跟`B`中对应位置的值相乘

`A .^ 2`代表将`A`中每个值平方

一般地，在运算符前加上`.`符号代表element-wise operation

`log(v)`代表将`v`中的每个值取对数，`exp(v)`代表将`v`中的每个值作为以$$e$$为底数的幂的次数

`abs(v)`可以获得`v`中每个值的绝对值，`-v`获得`v`中每个值的相反数

`A`的转置用`A'`表示

`max(a)`可以获得向量中的最大值，若使用两个变量去储存结果，则第一位是最大值，第二位是最大值的位置，例子如下

```matlab
a = [1 15 2 0.5]
val = max(a)  % val = 15
[val, ind] = max(a)  % val = 15, ind = 2
```

若`A`是矩阵，则用法如下

```matlab
A = [8 1 6; 3 5 7; 4 9 2]
max(A, [], 1)  % 结果为[8 9 7]（每一列的最大值）
max(A, [], 2)  % 结果为[8; 7; 9]（每一行的最大值）
```

`find(a < 3)`可以找出`a`中小于3的值的位置（`a < 3`是返回一个boolean vector），若`find(A < 3)`（`A`是矩阵），则用法如下列代码所示

```matlab
A = magic(3)  % 产生一个3 * 3的矩阵
[r, c] = find(A < 3)
```

其中`r`是符合条件的值的行数的集合，`c`是符合条件的值的列数的集合

`sum(a)`为求和，`prod(a)`是求每个值的积，`floor(a)`和`ceil(a)`为向下/上取整

若`A`为矩阵，`sum(A, 1)`为对矩阵`A`的行分别求和，`sum(A, 2)`是对每列分别求和，若要求对角线的和，可以使用`A .* eye(n)`（假设`A`是$$n\times n$$矩阵）的到一个只有对角线的矩阵，因此`sum(sum(A .* eye(n)))`即可求出`A`的对角线和，也就是$$\text{tr}A$$，矩阵$$A$$的迹

`flipud(A)`可以得到矩阵`A`垂直翻转后的矩阵

```matlab
A = [1 2; 3 4]
flipud(A)  % [3 4; 1 2]
```

### 绘制图像

`plot(x, y)`可以绘制图像，若要将多个$$y$$绘制到同一个图像中，在使用下一个`plot`命令前要先输入`hold on`来保持已有的图像

```matlab
x = [0:0.01:1];
y1 = sin(x);
y2 = cos(x);
plot(x, y1);
hold on;
plot(x, y2, 'r');  % 使用红色
```

`xlabel(label_text)`和`ylabel(label_text)`可以分别将横轴和纵轴的标题修改为`label_text`

`legend(name1, name2)`可以设置图像的图例

`title(plot_name)`可以设置图像的标题

`print -dpng file_name`可以将图像保存为图片

`close`命令可以关闭图像窗口

可以用`figure(n)`来指定不同的图像窗口

`subplot(1, 2, 1)`可以将图像分成$$1\times 2$$的网格，并返回第一个元素

`axis([x_min x_max y_min y_max])`可以按照所给参数重新缩放坐标轴

`clf`可以清除图像

`imagesc(A)`可以将矩阵`A`绘制为图像，每个值拥有自己的颜色

### 控制语句

#### `for`循环

```matlab
v = zeros(10, 1);  % 创建一个10维的零向量
for i = 1:10,
    v(i) = 2 ^ i;
end;  % v的值为[2; 4; 8; 16; 32; 64; 128; 256; 512; 1024]
```

#### `while`循环

```matlab
v = zeros(10, 1);
while i <= 5,
    v(i) = 100;
    i = i + 1;
end;
```

#### `if`、`else`、`break`和`continue`

```matlab
i = 1;
v = zeros(10, 1);
while true,
    v(i) = 999;
    i = i + 1
    if i == 6,
        break;
    end;
end;

random = rand();
if random == 0.5,
    disp('Serious?');
elseif random < 0.5,
    disp('random < 0.5');
else
    disp('random > 0.5');
end;
```

#### 函数

函数定义在`.m`文件中

假设我们有名为`testFunction.m1`的文件，内容如下

```matlab
function f = testFunction(x)

y = x ^ 2;
```

确保工作路径与文件所在路径相同，直接输入`testFunction(arg)`即可执行函数

函数可以返回多个值

```matlab
function [r1, r2] = getTwoRandN()

r1 = randn();
r2 = randn();
```

### Vectorization

通过将计算转换为向量之间的操作，可以减少循环在代码中的使用，从而使代码的运行效率提升

#### 非向量化实现$$h_\theta(x)=\sum_{j = 0}^n \theta_jx_j$$

```matlab
% 假设theta和x以向量形式储存
prediction = 0.0;
for j = 1:n+1,  % MATLAB是以1作为第一个index
    prediction = prediction + theta(j) * x(j)
end;
```

#### 向量化实现$$h_\theta(x)= \mathbf{\theta}^T\mathbf{x}$$

```matlab
prediction = theta' * x;  % 加一撇代表转置
```

#### 向量化实现梯度下降

$$\mathbf{\theta} := \mathbf{\theta} - \alpha\mathbf{\delta}$$，其中$$\mathbf{\theta}$$和$$\mathbf{\delta}$$为向量，$$\delta = \begin{bmatrix} \delta_0 \\ \vdots \\ \delta_n \end{bmatrix}$$，且对于$$j\in[0, n]$$，有$$\delta_j = {1\over m}\sum_{i=0}^mx_j^{(i)}(h_\theta(\mathbf{x}^{(i)})-y^{(i)})$$

此外，我们有$$\mathbf{x}^{(i)} = \begin{bmatrix} x_0^{(i)} \\ \vdots \\ x_n^{(i)} \end{bmatrix}$$

因此，$$\mathbf{\delta}={1\over m} \sum_{i=1}^m\mathbf{x}^{(i)}(h_\theta(\mathbf{x}^{(i)})-y^{(i)})$$

## Normal Equation的推导

### 矩阵求导（Matrix Derivatives）

对于一个输入为$$m\times n$$矩阵输出为实数的函数$$f:\mathbb{R}^{m\times n} \to \mathbb{R}$$，我们定义它关于矩阵$$A$$的导函数为

$$\nabla_Af(A)=\begin{bmatrix} \frac{\partial f}{\partial A_{11}} &\cdots &\frac{\partial f}{\partial A_{1n}} \\ \vdots &\ddots &\vdots \\ \frac{\partial f}{\partial A_{m1}} &\cdots &\frac{\partial f}{\partial A_{mn}} \end{bmatrix}$$

换句话说，梯度$$\nabla_Af(A)$$是一个$$m\times n$$的矩阵，它的在$$(i,j)$$位置上的值是$$\frac{\partial f}{\partial A_{ij}}$$。举个例子，假设$$A=\begin{bmatrix} A_{11} &A_{12} \\ A_{21} &A_{22} \end{bmatrix}$$，且函数$$f:\mathbb{R}^{2\times 2} \to \mathbb{R}$$等于

$$f(A)=\frac{3}{2}A_{11}+5A^2_{12}+A_{21}A_{22}$$

那么我们有

$$\nabla_Af(A) = \begin{bmatrix} \frac{3}{2} &10A_{12} \\ A_{22} &A_{21} \end{bmatrix}$$

此外，我们再引进一个叫**迹（trace）**的运算符，写作$$\mathrm{tr}$$。对于一个$$n\times n$$的方阵$$A$$，$$A$$的迹为它对角线上的值的和，即

$$\mathrm{tr}A=\sum_{i=1}^{n}A_{ii}$$

如果$$a$$是一个实数（$$1\times 1$$的矩阵），那么$$\mathrm{tr}a=a$$。

迹有很多性质。对于矩阵$$A$$和$$B$$，若$$AB$$是方阵，则$$\mathrm{tr}AB=\mathrm{tr}BA$$。由此可以推出下列结论

$$\mathrm{tr}ABC=\mathrm{tr}CAB=\mathrm{tr}BCA \\ \mathrm{tr}ABCD=\mathrm{tr}DABC=\mathrm{tr}CDAB=\mathrm{tr}BCDA$$

此外，迹还有以下性质。此处$$A$$和$$B$$为方阵，$$a$$为实数

$$
\begin{eqnarray*}
\mathrm{tr}A&=&\mathrm{tr}A^T \\
\mathrm{tr}(A+B)&=&\mathrm{tr}A+\mathrm{tr}B \\
\mathrm{tr}aA&=&a\mathrm{tr}A
\end{eqnarray*}
$$

引入了迹的概念后，矩阵导数有下列性质

$$
\begin{eqnarray*}
\nabla_A\mathrm{tr}AB&=&B^T \tag{1}\\
\nabla_{A^T}f(A)&=&(\nabla_Af(A))^T \tag{2}\\
\nabla_A\mathrm{tr}ABA^TC&=&CAB+C^TAB^T \tag{3}\\
\nabla_A|A|&=&|A|(A^{-1})^T \tag{4}
\end{eqnarray*}
$$

下面让我们简单说明一下第一条性质的含义。

假设我们有某个固定的矩阵$$B\in \mathbb{R}^{n\times m}$$，我们可以构建一个函数$$f:\mathbb{R}^{m\times n} \to \mathbb{R}$$，使得对于矩阵$$A\in \mathbb{R}^{m\times n}$$，有$$f(A)=\mathrm{tr}AB$$。

我们可以对$$f(A)$$应用我们的矩阵求导，来求得$$\nabla_Af(A)$$，$$\nabla_Af(A)$$同样是一个$$m\times n$$的矩阵。式子$$\text{(1)}$$说明了对于矩阵$$\nabla_Af(A)$$中第$$(i,j)$$项的值，与$$B^T$$中第$$(i,j)$$项中的值相同（换句话说，与$$B$$中第$$(j,i)$$项的值相同）。

译者注：这里通过例子的方式简单解释一下前两个式子是怎么得到的

对于式子$$\text{(1)}$$，假设$$A = \begin{bmatrix} A_{11} &A_{12} &A_{13} \\ A_{21} &A_{22} &A_{23} \end{bmatrix} \in \mathbb{R}^{2\times 3}$$，$$B = \begin{bmatrix} B_{11} &B_{12} \\ B_{21} &B_{22} \\ B_{31} &B_{32} \end{bmatrix} \in \mathbb{R}^{3\times 2}$$，

易得

$$AB = \begin{bmatrix}
{A_{11}B_{11}+A_{12}B_{21}+A_{13}B_{31}} &{A_{11}B_{12}+A_{12}B_{22}+A_{13}B_{32}} \\
{A_{21}B_{11}+A_{22}B_{21}+A_{23}B_{31}} &{A_{21}B_{12}+A_{22}B_{22}+A_{23}B_{32}}
\end{bmatrix}
$$

则

$$
\begin{eqnarray*}
f(A)&=&\mathrm{tr}AB \\
    &=&A_{11}B_{11}+A_{12}B_{21}+A_{13}B_{31}+A_{21}B_{12}+A_{22}B_{22}+A_{23}B_{32} \\
\end{eqnarray*}
$$

代入我们之前定义的矩阵求导法则，得（为了节省空间，只给出$$\nabla_Af(A)$$第一项的展开形式，你可以自行代入其余的值进行检查）

$$
\begin{eqnarray*}
\nabla_Af(A) &=&
\begin{bmatrix}
{\partial (A_{11}B_{11}+A_{12}B_{21}+A_{13}B_{31}+A_{21}B_{12}+A_{22}B_{22}+A_{23}B_{32})\over{\partial A_{11}}}
&{\partial f\over{\partial A_{12}}}
&{\partial f\over{\partial A_{13}}} \\
{\partial f\over{\partial A_{21}}}
&{\partial f\over{\partial A_{22}}}
&{\partial f\over{\partial A_{23}}}
\end{bmatrix} \\
&=&
\begin{bmatrix}
B_{11} &B_{21} &B_{31} \\
B_{12} &B_{22} &B_{32}
\end{bmatrix} \\
&=& B^T
\end{eqnarray*}
$$

对于第一项，我们是关于变量$$A_{11}$$求微分，那么$$f(A)$$中其它的不包含$$A_{11}$$的项关于$$A_{11}$$的偏微分就是$$0$$，而在包含$$A_{11}$$的项中，$$A_{11}$$的次数是一次，一次函数求导为常数，即$$B_{11}$$。其它项也一样，以此类推，最终得到$$\nabla_Af(A)=B^T$$。

对于式子$$\text{(2)}$$，我们可以令$$C=A^T$$

$$\begin{eqnarray*}
\nabla_{A^T}f(A)&=&\nabla_{A^T}\mathrm{tr}AB \\
&=&\nabla_C\mathrm{tr}C^TB \\
&=&\nabla_C\mathrm{tr}(C^TB)^T \\
&=&\nabla_C\mathrm{tr}B^TC \\
&=&\nabla_C\mathrm{tr}CB^T \\
&=&(B^T)^T \\
&=&B \\
&=&(\nabla_Af(A))^T
\end{eqnarray*}$$

式子$$\text{(3)}$$我本人没有证出来，但是我在网上找到了一份材料，["Some Important Properties for Matrix Calculus"](https://dawenl.github.io/files/mat_deriv.pdf) by Dawen Liang，里面有对式子$$\text{(3)}$$的证明。

### 寻找Cost Function$$J$$关于$$\mathbf{\theta}$$的偏导数

回到正题，在lecture中我们定义了

$$X=\begin{bmatrix} (\mathbf{x}^{(1)})^T \\ (\mathbf{x}^{(2)})^T \\ \vdots \\ (\mathbf{x}^{(m)})^T \end{bmatrix}, \vec{y}=\begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}$$

因为我们知道$$h_\mathbf{\theta}(\mathbf{x}^{(i)})=(\mathbf{x}^{(i)})^T\mathbf{\theta}$$（译者注：这里无论是$$(\mathbf{x}^{(i)})^T\mathbf{\theta}$$还是$$\mathbf{\theta}^T(\mathbf{x}^{(i)})$$都是正确的，因为$$\mathbf{x}^{(i)}$$和$$\mathbf{\theta}$$都是同维度的向量)，我们可以得到下面的式子

$$
X\mathbf{\theta}-\vec{y} =
\begin{bmatrix} (\mathbf{x}^{(1)})^T \\ \vdots \\ (\mathbf{x}^{(m)})^T \end{bmatrix} - \begin{bmatrix} y^{(1)} \\ \vdots \\ y^{(m)} \end{bmatrix}
= \begin{bmatrix}
h_\mathbf{\theta}(\mathbf{x}^{(1)})-y^{(1)} \\ \vdots \\ h_\mathbf{\theta}(\mathbf{x}^{(m)})-y^{(m)}
\end{bmatrix}
$$

此外，对于任意向量$$z$$，我们有$$z^Tz=\sum_iz_i^2$$

所以，我们可以得到

$${1\over2}(X\mathbf{\theta}-\vec{y})^T(X\mathbf{\theta}-\vec{y})={1\over2}\sum_{i=1}^{m}(h_\mathbf{\theta}(\mathbf{x}^{(i)})-y^{(i)})^2 = J(\mathbf{\theta})$$

（译者注：这里的$$J$$比lecture提供的少除了一个$$m$$，但是不影响我们的最终结果，因为最后我们也是令$$\nabla_\mathbf{\theta}J(\mathbf{\theta})=0$$）

最后，为了找到$$J$$的最小值，我们开始求$$J$$关于$$\mathbf{\theta}$$的导函数。联立式子$$\text{(2)}$$和$$\text{(3)}$$，我们可以得到

$$
\nabla_{A^T}\mathrm{tr}ABA^TC=B^TA^TC^T+BA^TC \tag{5}
$$

因此，我们得到（运用转置矩阵的性质和矩阵乘法的性质）

$$\begin{eqnarray*}
\nabla_\mathbf{\theta}J(\mathbf{\theta})
&=& \nabla_\mathbf{\theta}{1\over2}(X\mathbf{\theta}-\vec{y})^T(X\mathbf{\theta}-\vec{y}) \\
&=& {1\over2}\nabla_\mathbf{\theta}(\mathbf{\theta}^TX^TX\mathbf{\theta}-\mathbf{\theta}^TX^T\vec{y}-\vec{y}^TX\mathbf{\theta}+\vec{y}^T\vec{y}) \\
&=& {1\over2}\nabla_\mathbf{\theta}\mathrm{tr}(\mathbf{\theta}^TX^TX\mathbf{\theta}-\mathbf{\theta}^TX^T\vec{y}-\vec{y}^TX\mathbf{\theta}+\vec{y}^T\vec{y}) \\
&=& {1\over2}\nabla_\mathbf{\theta}(\mathrm{tr}\mathbf{\theta}^TX^TX\mathbf{\theta}-2\mathrm{tr}\vec{y}^TX\mathbf{\theta}) \\
&=& {1\over2}(X^TX\mathbf{\theta}+X^TX\mathbf{\theta}-2X^T\vec{y}) \\
&=& X^TX\mathbf{\theta}-X^T\vec{y}
\end{eqnarray*}$$

在第三步，我们使用了“实数的迹是它本身”这一条性质（译者注：$$\mathbf{\theta}^TX^TX\mathbf{\theta}$$，$$\mathbf{\theta}^TX^T\vec{y}$$，$$\vec{y}^TX\mathbf{\theta}$$，以及$$\vec{y}^T\vec{y}$$的结果都是实数，因为$$\mathbf{\theta}$$和$$\vec{y}$$是向量）；在第四步，我们利用了$$\mathrm{tr}A=\mathrm{tr}A^T$$这条性质；在第五步，我们利用了式子$$\text{(5)}$$（其中$$A^T=\mathbf{\theta}$$，$$B=B^T=X^TX$$，$$C=I$$）。

译者注：这是第四步没有跳步的版本

$$
\begin{eqnarray*}
&&{1\over2}\nabla_\mathbf{\theta}\mathrm{tr}(\mathbf{\theta}^TX^TX\mathbf{\theta}-\mathbf{\theta}^TX^T\vec{y}-\vec{y}^TX\mathbf{\theta}+\vec{y}^T\vec{y}) \\
&=& {1\over2}\nabla_\mathbf{\theta}(\mathrm{tr}\mathbf{\theta}^TX^TX\mathbf{\theta}-\mathrm{tr}\mathbf{\theta}^TX^T\vec{y}-\mathrm{tr}\vec{y}^TX\mathbf{\theta}+\mathrm{tr}\vec{y}^T\vec{y}) \\
&=& {1\over2}\nabla_\mathbf{\theta}(\mathrm{tr}\mathbf{\theta}^TX^TX\mathbf{\theta}-\mathrm{tr}\mathbf{\theta}^TX^T\vec{y}-\mathrm{tr}\vec{y}^TX\mathbf{\theta}) \\
&=& {1\over2}\nabla_\mathbf{\theta}(\mathrm{tr}\mathbf{\theta}^TX^TX\mathbf{\theta}-\mathrm{tr}(\mathbf{\theta}^TX^T\vec{y})^T-\mathrm{tr}\vec{y}^TX\mathbf{\theta}) \\
&=& {1\over2}\nabla_\mathbf{\theta}(\mathrm{tr}\mathbf{\theta}^TX^TX\mathbf{\theta}-2\mathrm{tr}\vec{y}^TX\mathbf{\theta})
\end{eqnarray*}
$$

最后的$$\vec{y}^T\vec{y}$$直接移走是因为这一项中没有变量$$\mathbf{\theta}$$，这部分的偏导数为$$0$$。

此外，上面这几步中用到的一些其它矩阵或迹的性质如下所示

$$
\begin{eqnarray*}
\mathrm{tr}(A+B)&=&\mathrm{tr}A+\mathrm{tr}B \\
(C+D)^T&=&C^T+D^T \\
(EF)^T&=&F^TE^T \\
(G+H)I&=&GI+GH \\
I(G+H)&=&IG+IH \\
(J^T)^T&=&J \\
\nabla_{K^T}f(K)&=&(\nabla_Kf(K))^T
\end{eqnarray*}
$$

最后，我们令$$\nabla_\mathbf{\theta}J(\mathbf{\theta})=0$$，即

$$
\begin{eqnarray*}
X^TX\mathbf{\theta}-X^T\vec{y} &=& 0 \\
X^TX\mathbf{\theta} &=& X^T\vec{y} \\
(X^TX)^{-1}X^TX\mathbf{\theta} &=& (X^TX)^{-1}X^T\vec{y} \\
\mathbf{\theta} &=& (X^TX)^{-1}X^T\vec{y}
\end{eqnarray*}
$$

*翻译自[Stanford CS229 Handouts](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf) by Andrew Ng from Stanford*
