---
layout: post
title:  "C++ 中 Lambda 表达式作为函数参数"
date:   2018-10-09 00:00:00 -0700
tags: study-cs cpp
---

C++ 自 C++11 起开始 Lambda 表达式（Lambda Expression），我一般简称为匿名函数（虽然这个称呼不太对）。一个匿名函数由三部分组成，方括号内的是 Capture，圆括号内的是 Parameter，花括号内的是函数体。下面是一个简单的匿名函数。

```cpp
auto foo = [x, &y](const T& z){ return x + y - z; }
```

一个匿名函数有三部分，首先是 `[]` 中的 capture 部分，我们在这里规定这个匿名函数要怎样处理外部变量。

在匿名函数中有两种 capture 的方法，一种是 capture by value ，另一种是 capture by reference。上面的例子中，`x` 就是一个 captured by value 的外部变量；而 `y` 是一个 captured by reference 的外部变量（注意 `y` 前面的符号 `&`，是 C++ 中表示引用的符号）。要注意，captured by reference 的外部变量在匿名函数里默认是常量，不能被修改（无论它本身是不是 `const`）。

`()` 中的是parameter，跟一般的函数一样。最后是 `{}` 中的 function body，也是跟一般的函数一样。

值得注意的是，Lambda 表达式虽然跟匿名函数很像，但是它并不是真正的“函数”，而是overload了 `operator()` 的一个匿名类的一个对象（所以这是为什么我说匿名函数这个名字不太对）。但是，在 C++11 中，对于没有 capture 外部变量的 Lambda 表达式，我们可以将它转型为函数指针。

```cpp
void (*foo)() = [](){};
int (*add)(int, int) = [](int a, int b){ return a + b; }
```

绝大多数情况下这种转型是没有问题的，甚至 `clang-tidy` 会提示我这种转型是多余的。但是，如果你是在配合模版使用 Lambda 表达式，你就很可能需要显式地告诉编译器我想把这个看做成一个函数。

举个例子，有如下代码。

```cpp
#include <iostream>

class Foo {
public:
    explicit Foo(void(*func)()) {
        std::cout << "using function pointer constructor" << std::endl;
    }

    template <class T> explicit Foo(T t) {
        std::cout << "using template constructor" << std::endl;
    }
};

int main() { Foo([](){}); return 0; }
```

上面这段代码的执行结果如下。

```text
using template constructor
```

原因正是我刚才所讲的，Lambda 表达式本质上是一个匿名类，自然而然编译器会选用带有模版的构造函数。如果我们确实需要调用函数指针的那一个构造函数，我们就需要一些处理。

```cpp
using EmptyFunction = void(*)();

// 方法1
Foo((EmptyFunction) [](){});

// 方法2
Foo(EmptyFunction{[](){}});
```

方法1是通过 C 风格转型；方法2是通过 C++ 的直接列表初始化实现“转型”。
