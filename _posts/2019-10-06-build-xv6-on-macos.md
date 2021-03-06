---
layout: post
title:  "在 macOS 上安装 xv6"
date:   2019-10-06 13:27:00 -0700
tags:   study-cs operating-system
---

## *2019-12-18 更新*

貌似 Homebrew 的 i386-elf-binutils 和 i386-elf-gcc 出了点问题，编译出来的 xv6 系统会导致 QEMU 卡在启动环节。

## 前言

xv6 是用 C 语言实现的 Unix Version 6 教学用操作系统，由 MIT 的教职工维护。

[xv6 官方主页](https://pdos.csail.mit.edu/6.828/2019/xv6.html)

[xv6 GitHub 仓库地址](https://github.com/mit-pdos/xv6-public)

[xv6 中文手册](https://th0ar.gitbooks.io/xv6-chinese/content/)

我在根据官方主页上的文档安装时，发现上面的说明和实际的 `Makefile` 不一致，于是自己尝试，最后完成了 xv6 的安装。这里建议先遵循官方的文档的说明进行安装，若失败，再尝试按照本文进行安装。我的 macOS 版本是 10.14.6。

## 安装依赖项

首先，确保你的 macOS 上已经安装了 [Homebrew](https://brew.sh/index_zh-cn)。

打开终端，输入

```bash
brew install i386-elf-gcc
```

使 Homebrew 安装 i386-elf-binutils 和 i386-elf-gcc。

然后输入

```bash
brew install qemu
```

使 Homebrew 安装 [QEMU](https://www.qemu.org)。

## 编译 xv6

下载 xv6 的源代码并进入源代码所在的目录，然后使用 `make` 命令进行编译。

```bash
git clone https://github.com/mit-pdos/xv6-public.git && cd xv6-public && make TOOLPREFIX="i386-elf-"
```

如果没有报错，则 xv6 编译完成。

## 从 QEMU 运行 xv6

在 xv6 的根目录下运行

```bash
make TOOLPREFIX="i386-elf-" qemu
```

出现下图所示窗口时代表 xv6 已经成功在 QEMU 中运行了。

![image-20191006132110088](/assets/2019-10-06-build-xv6-on-macos/image-20191006132110088.png)

如果不想启用 VGA 输出，则可以使用

```bash
make TOOLPREFIX="i386-elf-" qemu-nox
```

来启动 xv6。

### 退出 QEMU

按下 `^A+X` 组合键即可退出。
