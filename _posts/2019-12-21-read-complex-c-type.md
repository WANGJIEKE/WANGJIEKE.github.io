---
layout: post
title:  "如何读懂 C 语言中变量的类型"
date:   2019-12-21 15:07:00 -0700
tags:   study-cs operating-system
---

先来一个小测试。请说出下列变量的类型

```c
int a;
int *b;
int *c();
int d[2];
int *e[2];
int (*f)[2];
int (*g)();
int *(*h)();
int *(*i[2])();
int *(*(**j[2][2])(char))[];
```

是不是有点难？

## `int* p` 与 `int *p`

允许我先跑个题，讲讲 `int *p` 与 `int* p` 这两种写法到底有什么不同。这两种写法都是符合 C/C++ 语言标准的，都是正确的语法。这两种写法的区别在于，它们强调的地方不一样。

[根据 C++ 之父 Bjarne Stroustrup 的说法](http://www.stroustrup.com/bs_faq2.html#whitespace)，一般来讲，“典型的”的 C++ 程序员偏向于使用 `int* p`。这种写法强调了**变量 `p` 的类型是 `int*`**，即一个指向 `int` 的指针。

反之，“典型的” C 程序员偏向于 `int *p`，并且把它解读为“变量 `p` 经过 `*` 的‘解引用’（dereference）操作后，会得到一个 `int`”，即强调 **`*p` 的结果是一个 `int`**”。

因此，指针类型里的 `*` 号并不是空穴来风，而是基于类似“解方程”的思想得到的。这里再举个例子，如果我想表示“变量 `q` 是一个 pointer to pointer to `int`”（翻译成中文是“指向‘指向 `int` 的指针’的指针”），也就是说，变量 `q` 要经过两次 `*` 的“解引用”操作，才能得到一个 `int` 类型的值。因此，我们变量 `q` 声明为 `int **q`。

## “运算符”优先级

在弄明白指针类型里的 `*` 是怎么来的之后，我们再来复习一下，C 语言中“运算符”的优先级。我们这里只关注三个会在类型中出现的“运算符”，`*`、`[]`、`()`。

| 优先级 | “运算符”       | 描述           | 结合性   |
| ------ | -------------- | -------------- | -------- |
| 1      | `()`<br />`[]` | 函数<br />数组 | 从左往右 |
| 2      | `*`            | 解引用         | 从右往左 |

这里的“运算符”加了双引号的原因是，这些符号并不是真正意义上的运算符，它们实际上是声明符（declarator），具体可以参考 [cppreference](https://en.cppreference.com/w/c/language/declarations)。

话虽如此，我们完全可以把它们当成运算符，来方便我们理解。

最后回顾一下顺序。先按照优先级，优先级高的先算；优先级相同时，按照结合性的方向计算。

## C 变量声明的组成

所有的 C 变量声明都由两部分组成，位于最左侧的*最终类型*，以及位于右侧，含有变量名和 `*`、`[]`、`()` 符号的*式子*。

左侧的最终类型一定是 C 中已有的类型（包括 `struct`、`union`、`enum`）。左侧的最终类型甚至可以是 `void`（但是你不能声明 `void a` 或者 `void a[]`，这两个是例外）。

右侧的式子则是告诉我们，对变量名进行何种操作，就可以获得左侧的最终类型的数据。

## 一个简单的例子

C 语言 `main` 函数有两个参数，`int argc` 和 `char *argv[]`。这里我就来讲解一下如何按照前面提到的“解方程”思想和运算符优先级，来理解 `argv` 的类型。我会将计算完毕的部分用 `_` 替换。

首先观察最左侧，发现类型是 `char`。说明 `argv` 的最终类型是 `char`，即“**`argv` is ... `char`**”。

再观察 `argv` 周围，左侧是 `*`，右侧是 `[]`。根据优先级，我们先看 `argv[]`，即“`argv` is **an array of** ... `char`”。此时原声明化简为

`argv[]` 之后，轮到左侧的 `*`，得到“`argv` is an array of **pointer to** ... `char`”。

`*argv[]` 之后，发现右侧的式子解析完毕，说明参数类型已经分析完毕。因此，最终 `argv` 的类型为“`argv` is an array of pointer to `char`”。翻译成中文是“`argv` 是一个由指向 `char` 的指针构成的数组”。

## 另一个简单的例子

来看这个例子

```c
long **foo[7];
```

同样，先观察最终类型，发现是 `long`，即“**`foo` is ... `long`**”。

然后根据优先级，先看 `foo[7]`，即“`foo` is **an array of 7** ... `long`”。

之后，因为整个声明只剩下左侧两个 `*` 号，所以最终的结果为“`foo` is an array of 7 **pointer to pointer to** `long`”。翻译成中文是“`foo` 是一个，由指向‘指向 `long` 的指针’的指针构成的，长度为 7 的数组”。

## 稍微难一点的例子

我们最后来看看文章开头里给出的这个声明

```c
int *(*(**foobar[][3])(char))[];
```

最终类型为 `int`，因此“**`foobar` is ... `int`**”。

观察变量名 `foobar` 周围，左侧是 `*`，右侧是 `[2]`。根据优先级，先往右边走，得到“`foobar` is **an array of** ... `int`”。

同样根据优先级，因为右侧有 `[3]` 存在，我们继续往右，得到“`foobar` is an array of **array of 3** ... `int`”。

解决完 `foobar[2][3]` 这部分，接下来要处理它左侧的两个 `*` 号，得到“`foobar` is an array of array of 3 **pointer to pointer to** ... `int`”。

目前为止，我们已经处理完了 `**foobar[2][3]` 这部分。如果我们将“计算完毕”的部分用 `_` 替换，原声明看起来就会像是这个样子

```c
int *(*(_)(char))[];
```

因为我们“计算”完了括号里的部分，我们可以顺便把括号一并拿走。

```c
int *(*_(char))[];
```

根据优先级，因为右侧是函数的符号 `()`，我们应该先往右侧走。函数符号 `()` 的内部是函数参数的类型，因此，我们目前得到的结果是“`foobar` is an array of array of 3 pointer to pointer to **function taking `char` returning** ... `int`”。此时剩下的部分是

```c
int *(*_)[];
```

处理括号里仅存的 `*`，得到“`foobar` is an array of array of 3 pointer to pointer to function taking `char` returning **pointer to** ... `int`”。同样，替换并去括号，得

```c
int *_[];
```

剩下的就很简单了，根据优先级，先往右处理 `[]`，最后处理左侧的 `*`。最终结果是“`foobar` is an array of array of 3 pointer to pointer to function taking `char` returning pointer to **array of pointer to** `int`”。翻译成中文是…… 算了还是不翻译了。

## 引用

[Reading C type declarations](http://unixwiz.net/techtips/reading-cdecl.html)

[cdecl: C gibberish ↔ English](https://cdecl.org/)
