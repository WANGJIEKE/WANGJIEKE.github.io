---
layout: post
title:  "如何读懂 C 语言中变量的类型"
date:   2020-01-20 01:11:00 -0700
tags:   study-cs c
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

## 方法

这里有一个较真起来可能有错误，但是可行的理解方式，就是**把 C 语言变量类型也看成一种“运算”**。

举个例子，当你看到 `int *p` 的时候，应该先把它理解为**“变量 `p` 经过运算符 `*` 的 dereference 操作之后得到的是一个 `int`”**。

此外，这个“运算”也遵守 C 语言运算符优先级规则（即 function call `()` 和 array indexing `[]` 运算符的优先级高于 dereference `*` 运算符，然后有括号先算括号里面的）。举个上面出现过的例子。对于 `int *c()` 来说，由于 `()` 运算符的优先级高于 `*` 运算符，所以我们应该先让变量 `c` 与 `()` 运算符进行运算，得到的结果再与 `*` 运算符进行运算。换句话说，就是**“对函数 `c` 的返回值进行 dereference 操作之后得到的是一个 `int`”**。

类似地，`int (*g)()` 则是**“对变量 `g` 进行 dereference 操作之后得到一个函数，这个函数的返回类型是 `int` ”**。

## 规则

1. 高优先级的 `()` 和 `[]` 运算符都是右结合的，而低优先级的 `*` 运算符是左结合的。我们可以将这个规律总结为，**只要能往右走就往右走，到无路可走时往左走**。
2. - 将 `x(params)` 翻译成**“`x` is a function taking `params` returning ...”**；
     - 如果函数没有参数，即 `x()`，则翻译成**“`x` is a function returning ...”**；
   - 将 `x[]` 翻译成**“`x` is an array of ...”**；
     - 如果有数字，即 `x[len]`，则翻译成**“`x` is an array (length is `len`) of ...”**；
   - 将 `*x` 翻译成**“`x` is a pointer to ...”**；
   - 注意这里的 `x` 不仅仅代表变量名，还可以代表变量名与运算符进行任意次“运算”后的结果。

## 例子

现在让我们实际运用一下上面的两个规则。这里的例子是上面出现过的 `int *(*i[2])()`。

运用规则 1，先往右边走；再运用规则 2，得到翻译“`i` is an array (length is 2) of ...”。如果我用 `_` 替换掉已经运算完成的部分，则剩下的类型声明是 `int *(*_)()`。

根据规则 1，无法往右走时往左走（因为我们必须先算括号内的部分，而此时 `_` 的右侧已经没有任何运算符了），然后根据规则 2，得到翻译“`_` is a pointer to ...”。代入前面得到的部分（用斜体表示），得“`i` is *an array (length is 2) of* pointer to ...”。此时剩下的类型声明时 `int *(_)()`。因为括号内部没有运算了，所以我们可以将括号去掉，得到 `int *_()`。

运用规则 1，继续往右侧走。运用规则 2，得到翻译“`_` is a function returning ...”。代入前面的结果，得“`i` is *an array (length is 2) of pointer to* a function returning ...”。剩余的类型声明为 `int *_`。

根据规则 1 往左走，根据规则 2 得到翻译“`_` is a pointer to `int`”。代入前面的结果，得到“`i` is *an array (length is 2) of pointer to a function returning* a pointer to `int`”。

至此，我们就得到了变量 `i` 的类型。

## 练习

仿照着前面的例子，尝试解读下面的 C 变量类型。

```c
int *(*(**j[2][2])(char))[];
const int * const(* const(* const * const k[2][2])(const char))[];
```

你可以到 [cdecl: C gibberish ↔ English](https://cdecl.org/) 验证你的答案。

## 参考资料

[Reading C type declarations](http://unixwiz.net/techtips/reading-cdecl.html)

[cdecl: C gibberish ↔ English](https://cdecl.org/)
