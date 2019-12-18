---
layout: post
title:  "在 Markdown 中使用 LaTeX 语法书写数学公式"
date:   2019-08-11 13:59:00 -0700
tags:   study-cs 
---

**本文章的重点是关于如何在 Markdown 文档中使用 $$\LaTeX$$ 语法书写数学公式，因此本文章并不会介绍如何使用 $$\LaTeX$$ 进行排版等与 $$\LaTeX$$ 本身有关的内容。此外，标准的 Markdown 中是不包含 $$\LaTeX$$ 数学公式的，这只是一种常见的 Markdown 扩展功能，有很多编辑器或者阅读器支持而已。**

## 行内数学公式

很多 Markdown 编辑器支持用单个美元符号（`$`）或者两个美元符号（`$$`）来包裹行内数学公式。下面是一个例子。

```markdown
快速排序的时间复杂度是 $$O(n\logn)$$，最坏情况下是 $$O(n^2)$$。
```

快速排序的时间复杂度是 $$O(n\log n)$$，最坏情况下是 $$O(n^2)$$。

## 数学公式块

一般的 Markdown 编辑器中，用两个单独在一行的美元符号来包括块状数学公式，注意美元符号和文本之间也要隔一行。下面是一个例子。

```markdown
下面是一个二次函数（假设 $$a \ne 0$$）

$$
y = ax^2 + bx + c
$$

```

下面是一个二次函数（假设 $$a \ne 0$$）

$$
y = ax^2 + bx + c
$$

## 常用 $$\LaTeX$$ 语法

### 上标下标

| $$\LaTeX$$              | 渲染后                      |
| ----------------------- | -------------------------- |
| `x^2`                   | $$x^2$$                    |
| `a_n=5n`                | $$a_n=5n$$                 |
| `sum_{j=1}^n\theta_j^2` | $$\sum_{j=1}^n\theta_j^2$$ |

### $$n$$ 次方根

```latex
$$
\sqrt{x^2+y^2}+\sqrt[3]{z}
$$
```

$$
\sqrt{x^2+y^2}+\sqrt[3]{z}
$$

### 分数

```latex
$$
x_{1,2}=\frac{-b\pm\sqrt{b^2-4ac}}{2a}
$$
```

$$
x_{1,2}=\frac{-b\pm\sqrt{b^2-4ac}}{2a}
$$

### 重音符号

| $$\LaTeX$$ | 渲染后       |
| ---------- | ----------- |
| `\hat{i}`  | $$\hat{i}$$ |
| `\vec{a}`  | $$\vec{a}$$ |
| `\bar{x}`  | $$\bar{x}$$ |

### 其它常用符号

| $$\LaTeX$$                                                   | 渲染后                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `\sum_{i=0}^n`                                               | $$\sum_{i=0}^n$$                                             |
| `\int_{-\infty}^{+\infty}`                                   | $$\int_{-\infty}^{+\infty}$$                                 |
| `\prod_{j=0}^m`                                              | $$\prod_{j=0}^m$$                                            |
| `n \pm 3`                                                    | $$n \pm 3$$                                                  |
| `a \cdot b`、`5 \times 3`、`\div`                            | $$a \cdot b$$、$$5\times3$$、$$\div$$                        |
| `\ge`、`\le`、`\neq`、`\approx`、`\equiv`                    | $$\ge$$、$$\le$$、$$\neq$$、$$\approx$$、$$\equiv$$          |
| `\subset`、`\subseteq`、`\in`、`\supset`、`\supseteq`、`\owns` | $$\subset$$、$$\subseteq$$、$$\in$$、$$\supset$$、$$\supseteq$$、$$\owns$$ |
| `\cap`、`\cup`、`\land`、`\lor`、`\neg`                      | $$\cap$$、$$\cup$$、$$\land$$、$$\lor$$、$$\neg$$            |
| `\gets`、`\to`、`\leftrightarrow`                            | $$\gets$$、$$\to$$、$$\leftrightarrow$$                      |
| `\nabla`、`\forall`、`\exists`、`\partial`                   | $$\nabla$$、$$\forall$$、$$\exists$$、$$\partial$$           |
| `\cdots`                                                     | $$\cdots$$                                                   |

### 数学字母

| $$\LaTeX$$          | 渲染后                |
| ------------------- | --------------------- |
| `\mathcal{ABCxyz}`  | $$\mathcal{ABCxyz}$$  |
| `\mathfrak{ABCxyz}` | $$\mathfrak{ABCxyz}$$ |
| `\mathbb{ABCxyz}`   | $$\mathbb{ABCxyz}$$   |
| `\mathbf{ABCxyz}`   | $$\mathbf{ABCxyz}$$   |
| `\text{ABCxyz}`     | $$\text{ABCxyz}$$     |

### 矩阵

对于类似矩阵这种天然多行的元素，在 $$\LaTeX$$ 中我们需要将它用 `\begin{bmatrix}\end{bmatrix}` 包起来。`bmatrix` 代表这是一个用方括号表示的矩阵（**B**racket **Matrix**）。里面的 `&` 符号可以调整对齐的方式，让矩阵更好看。

```latex
$$
I_n = \begin{bmatrix}
1& 0& \cdots& 0 \\
0& 1& \cdots& 0 \\
\vdots& \vdots& \ddots& \vdots \\
0& 0& \cdots& 1
\end{bmatrix}
$$
```

$$
I_n = \begin{bmatrix}
1& 0& \cdots& 0 \\
0& 1& \cdots& 0 \\
\vdots& \vdots& \ddots& \vdots \\
0& 0& \cdots& 1
\end{bmatrix}
$$

### 方程组

类似于矩阵，我们使用 `eqnarray*` 表示一个方程组。同样我们可以通过设置 `&` 来让方程组里的方程在等号处对齐。

```latex
$$
\begin{eqnarray*}
y &=& 4x + 3 \\
y &=& 2x^2 + 5x - 6
\end{eqnarray*}
$$
```

$$
\begin{eqnarray*}
y &=& 4x + 3 \\
y &=& 2x^2 + 5x - 6
\end{eqnarray*}
$$

### 分段函数

类似于矩阵，我们使用 `cases` 表示一个分段函数。其中，`\quad` 代表空格；如果需要更长的空格，可以用 `\qquad`。

```latex
$$
f(x) = \begin{cases}
1\quad\text{(if }0\le x\le1 \text{)}\\
0\quad\text{(otherwise)}
\end{cases}
$$
```

$$
f(x) = \begin{cases}
1\quad\text{(if }0\le x\le1 \text{)}\\
0\quad\text{(otherwise)}
\end{cases}
$$

## 参考资料

- [常用数学符号的 LaTeX 表示方法](http://mohu.org/info/symbols/symbols.htm)
- [LaTeX 编写分段函数](https://blog.csdn.net/u012428169/article/details/76422845)
