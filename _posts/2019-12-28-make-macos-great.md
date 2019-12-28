---
layout: post
title:  "如何调教 macOS 系统"
date:   2019-12-28 13:44:00 -0700
tags:   macOS
---

## 优化 Dock

在 macOS 里默认的 Dock 位置是常驻屏幕的底端，不过我个人不喜欢默认的这个设计，因为这样子 Dock 会挡住很大一块屏幕。我的做法是将 Dock 移动到屏幕的左侧，并启用“Automatically hide and show the Dock”选项，如下图所示。

![image-20191228122304154](/assets/2019-12-28-make-macos-great/image-20191228122304154.png)

但是这样还是有个小问题，就是 Dock 默认的触发延迟和动画时间都太长了。鼠标移动到最左侧之后要等上一阵子才能看到 Dock 慢吞吞地显示出来。不过幸好，我们可以自行修改触发延迟和动画时间。

```bash
# 让鼠标移动到左侧时立刻显示 Dock
defaults write com.apple.dock autohide-delay -int 0 && killall Dock

# 加快 Dock 的动画速度（数值可以根据喜好自行调整）
defaults write com.apple.dock autohide-time-modifier -float 0.4 && killall Dock
```

## 优化 Mission Control

这是我的 Mission Control 设置，仅供参考。

![image-20191228125140685](/assets/2019-12-28-make-macos-great/image-20191228125140685.png)

这里的重点是左下角“Hot Corners…”的设置。

![image-20191228125256072](/assets/2019-12-28-make-macos-great/image-20191228125256072.png)

我将左上角设置为锁屏，左下角设置为睡眠，右下角设置为显示桌面（与 Windows 10 中右下角为显示桌面按钮保持一致）。

## 修改鼠标滚轮方向

可能很多小伙伴跟我一样，觉得 macOS 里鼠标滚轮的方向“不太对劲”。不扯什么设计哲学，如果你跟我一样，不想改变自己的习惯，或者跟我一样是同时使用 Windows 和 macOS 的人，那么有一个叫 [Scroll Reverser](https://pilotmoon.com/scrollreverser/) 的开源小程序能帮你解决这个问题。

![image-20191228130137778](/assets/2019-12-28-make-macos-great/image-20191228130137778.png)

注意，鼠标设置里也要选中“Scroll direction: Natural”。

![image-20191228130534541](/assets/2019-12-28-make-macos-great/image-20191228130534541.png)

## 小工具

下面介绍一些能让 macOS 更易用的小程序。

### [BetterSnapTool](https://apps.apple.com/us/app/bettersnaptool/id417375580)

怀念 Windows 10 里将窗口拖到顶部就能最大化，拖到右侧就能让窗口宽度设置为屏幕宽度一半的功能吗（别告诉我你不知道 Windows 10 里有个功能）？有一个叫 BetterSnapTool 的程序能帮你在 macOS 里做到这一点。

![Screen Shot 2019-12-28 at 1.08.26 PM](/assets/2019-12-28-make-macos-great/Screen Shot 2019-12-28 at 1.08.26 PM.png)

### [Alfred](https://www.alfredapp.com/)

这是一个非常方便的搜索 app。我个人是将快捷键设定为 ⌥Space （⌃Space 是切换输入法，⌘Space 是打开 macOS 自带的 Spotlight）。

![image-20191228132523595](/assets/2019-12-28-make-macos-great/image-20191228132523595.png)

### [Snipaste](https://zh.snipaste.com/)

这是一个非常高效的截图/贴图软件。尤其是在写代码的时候，可以将文档中的示例代码截图，然后使用 Snipaste 的贴图功能贴在屏幕上，这样就不需要在浏览器和文本编辑器之间来回切换了（这个软件也有 Windows 版）。

![image-20191228133051874](/assets/2019-12-28-make-macos-great/image-20191228133051874.png)

### [iTerm 2](https://iterm2.com/)

非常强大的一个终端模拟器。

![image-20191228133152062](/assets/2019-12-28-make-macos-great/image-20191228133152062.png)

### [Proxifier](https://www.proxifier.com/)

有些 app（比如 macOS 自带的 Mail）会无视我们设置代理，依然强行使用直连。我们可以用 Proxifier 来强制让这些 app 的流量通过代理。

### [Hidden Bar](https://apps.apple.com/app/hidden-bar/id1452453066)

安装完这些程序之后发现顶栏塞满了图标，空间不够用了？Hidden Bar 这个小程序可以帮你隐藏不常用的图标。

![image-20191228134249054](/assets/2019-12-28-make-macos-great/image-20191228134249054.png)

![image-20191228134259740](/assets/2019-12-28-make-macos-great/image-20191228134259740.png)
