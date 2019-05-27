---
layout: post
title:  "设置ssh的config文件以实现快速访问"
date:   2019-05-26 17:50:00 -0700
tags: study-cs linux
---

*本文章内容参考自Stack Overflow上的[这个问答](https://stackoverflow.com/a/2419609/9525608)。*

在本博客的[第一篇文章]({% post_url 2018-07-12-ssh-key %})介绍如何使用密钥进行ssh连接的验证之后，可能有些人会碰到像我一样的问题。平常管理的服务器有点多，每次输入完整的用户名、服务器地址和密钥路径太烦了。有没有什么办法能够自动管理这些信息呢？

对于这个，我们只需要在`~/.ssh/`文件夹中，创建一个名叫`config`的文件。

```bash
cd ~/.ssh
touch config
```

打开`config`文件，将以下模版复制到文件中，按照自己的实际情况修改即可。

```text
# my_short_name是自己定义的简写名，real_name.com是服务器的地址
Host my_short_name real_name.com

    # real_name.com是服务器的地址（跟上面的一致）
    HostName real_name.com

    # ~/.ssh/key是用于连接的私钥的路径
    IdentityFile ~/.ssh/key

    # user_name是用户名
    User user_name
```

保存修改之后，下次再要进行ssh连接时，输入

```bash
ssh my_short_name
```

（等价于）

```bash
ssh -i ~/.ssh/key user_name@real_name.com
```

即可完成连接。
