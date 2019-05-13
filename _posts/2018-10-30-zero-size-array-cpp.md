---
layout: post
title:  "C++中new一个长度为0的数组"
date:   2018-10-30 00:00:00 -0700
tags: study-cs cpp
---

最近看到老师的示例代码，其中有一个操作，就是`new`一个长度为`0`的数组，大概是下面这种样子。

{% highlight cpp %}
class Foo {
    int size = 0;
    int* arr;
    Foo() {
        arr = new int[size];
    }
};

{% endhighlight %}

我看完之后觉得非常神奇，`new`一个长度为`0`的数组，感觉不太符合常理。但是，经过一番搜索，我发现，这个确实是合法的操作。

首先，对于默认的`new[]`，它最简单的function signature是这样的。
{% highlight cpp %}
void* operator new[](std::size_t size);
{% endhighlight %}
注意到这个`size`的类型是`std::size_t`，这个类型是用来表示“某个对象的大小”，所以显然这是一个非负整数（`unsigned`），所以传入0肯定是没有问题的。

此外，因为`new int[0]`是合法的操作，自然而然`delete[] (new int[0])`也是合法的操作。

但是，我们并不能够使用这个数组。因为数组的长度为0，我们对数组进行的任何改动都会因为越界而变成未定义行为。
{% highlight cpp %}
int* array = new int[0];

// 这些都是未定义行为
array[0] = 10;
*array = 10;
{% endhighlight %}