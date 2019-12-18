---
layout: post
title:  "ASCII 动画生成与播放器"
date:   2019-08-28 15:01:00 -0700
tags:   study-cs python
---

## 简介

差不多两年前，我在 B 站发了一个 ASCII 版的[小埋体操](https://www.bilibili.com/video/av16719581)，当时是用 Python 配合（忘了是什么库）写的（代码已经找不到了）。

![image-20190827222535697](/assets/2019-08-28-ascii-animation/image-20190827222535697.png)

这个版本的小埋体操有几个问题。首先，它的播放速度是计算机执行循环的速度，而不是原视频的速度。所以在录制成视频之后，要我本人手动降速才能与原视频配上。此外，因为它的播放速度不是原视频的速度，以及一些其它原因（说白了就是太菜）所以导致无法播放原视频的音频。还有很重要的一点，在这个版本里，在显示每一帧的时候，我先调用了 `os.system('clear')`，然后才进行 `print`，这导致了整个终端窗口在一段很短的时间里是空白的，容易导致闪屏，降低了观看的体验。

最近忙里偷闲，从头撸了一套代码，从 ASCII 动画生成到播放可以一键（？）完成，甚至在 Windows 和 macOS 上还支持音频输出（因为我手头上的 Linux 设备都没有音频输出所以我也检测不了代码到底写没写对，干脆不写了），同时也解决了闪屏的问题。项目已经开源到 [GitHub](https://github.com/WANGJIEKE/ascii-animation) 上了。

## 实现视频到 ASCII 字符的转换

### 提取音频

其实从视频中提取音频非常容易，用 [FFmpeg](https://ffmpeg.org) 的话一行命令即可。不过要注意的是，FFmpeg 并不是 Python 中的模组，而是一个著名的可以对多媒体内容进行处理的工具和库。在 Python 中如果想要使用 FFmpeg 的话，最简单的办法是使用自带的 `subprocess` 模组，生成一个新的进程来执行 FFmpeg。对应的 FFmpeg 命令如下

```shell
ffmpeg -i "input_video.mp4" -map a:0 "output_audio.wav"
```

这里在输入和输出文件的路径外面加双引号的目的是为了处理含有特殊字符的路径（尤其是 Windows 系统下）。这里的 `-map a:0` 选项是代表只保留声轨的第零轨。

#### 用 `subprocess` 模组运行新进程

用 `subprocess` 模组创建进程非常简单，只需要调用 `subprocess.Popen` 这个类，然后往里面传对应的参数就好了。比如说对于上面的命令，我们可以这么调用 `Popen`

```python
import subprocess as sp  # 后面的代码块中都将subprocess简写为sp

p = sp.Popen(['ffmpeg', '-i', 'input_video.mp4', '-map', 'a:0', 'output_audio.wav'])
```

不过这个时候 Python 会继续运行下去，如果我们希望 Python 停下来等待 `ffmpeg`，我们可以调用 `p.wait()` 来达到这个目的。

#### 用 `shlex` 来处理命令行参数

`shlex` 是另一个功能强大的 Python 自带模组。它里面的 `split` 与 `str.split` 很类似，都可以帮助我们将字符串拆分。但是 `shlex.split` 与 `str.split` 不同的地方是，前者能识别出双引号，并且不会对双引号中的空格进行分割，特别适合处理命令行参数。

```python
import shlex
s = '1 "2 3" 4 5'
print(shlex.split(s))  # 结果是['1', '2 3', '4', '5']
print(s.split())       # 结果是['1', '"2', '3"', '4', '5']
```

将 `shlex` 和 `subprocess` 结合到一起的话，能增加代码的整体可读性

```python
def extract_sound(input_file, output_file):
    cmd = f'ffmpeg -i "{input_file}" -map a:0 "{output_file}"'
    p = sp.Popen(shlex.split(cmd))
    p.wait()
```

### 获得视频的帧率

可以通过 FFmpeg 中的 `ffprob` 获得视频的帧率。命令如下（同样用 `subprocess` 模组来生成新的进程以运行 `ffprob`）

```shell
ffprobe -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate "input.mp4"
```

给 `ffprob` 加上这些迷の参数（？）之后，输出就会变成只有一个代表帧率的数字。问题来了，这个数字是传输到标准输出里的，我们在 Python 中怎么获得到这个输出呢？其实也非常简单，万能的 `subprocess` 模组准备了一个叫 `subprocess.PIPE` 的神器，我们可以通过它来获得一个进程的标准输出和标准错误输出。

```python
def get_frame_rate(input_file):
    cmd = 'ffprob -of csv=p=0 -select_streams v:0 ' \
        f'-show_entries stream=r_frame_rate "{input_file}"'
    p = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    p.wait()
    return float(p.communicate()[0])
```

当我们将 `stdout` 参数指定为 `sp.PIPE` 时，我们可以通过 `p.communicate()[0]` 来获得标准输出的内容（顺便一提，如果我们将 `stderr` 指定为 `sp.PIPE`，我们可以通过 `p.communicate()[1]` 来获取标准错误输出的内容）。因为 `ffprob` 返回的是一个除法算式，我这里用 `eval` 把它转换成对应的 Python 数字类型，便于之后的处理。

### 改变视频的帧率

这个也很简单，就是通过 FFmpeg 将视频以目标帧率重新渲染一遍。

```shell
ffmpeg -i "input_file.flv" -filter:v fps=fps=target_fps "output.mp4"
```

这里有个坑就是输出格式尽量使用 `.mp4`。我在改变 `.flv` 文件的帧率时发现，如果输出文件的格式也是 `.flv`，视频的比特率会被大幅降低，输出的视频质量很差。如果输出文件的格式是 `.mp4` 的话就没这个问题。这个应该是 FFmpeg 针对不同格式的默认配置不同。

### 提取视频的所有帧

同样，一条 FFmpeg 命令就能搞定。

```shell
ffmpeg -i "input.mp4" -vf scale=-1:168, format=gray "output_dir/file_name_%05d.png"
```

这里的 `scale=-1:168` 代表将输出的图片的高度设为 168 像素，宽度为 -1 代表自动设置宽度并保持长宽比。`format=gray` 代表输出的帧的颜色为灰度。最后的输出目录中的 `%05d` 代表将图片用五位数进行编号（比如说 `output_dir/file_name_00001.png` 是 `input.mp4` 的第一帧经过缩放和颜色调整后的图片），编号不足五位数字的部分用 0 补齐。在经过上面的转换后，原视频 `input.mp4` 的每一帧都会被提取成高度为 168 像素，颜色为灰度的图片了。提取出来的图片的高度应该等于终端窗口高度减去一，这样能最大化字符的数量（拼出来的图形更清晰）。

### 将帧转换为 ASCII 字符

最后一步就是将所有图片帧转换成 ASCII 字符。首先考虑如何实现单个像素的灰度值到 ASCII 字符的映射。单个像素的灰度值的范围是 $$[0, 255]$$，然后我们使用的 ASCII 字符灰度映射有 10 个等级（如下所示）

```python
"@%#*+=-:. "  # 10个等级，从暗到亮（用于白底黑字）
```

也就是说索引的取值范围是 $$[0, 9]$$，因此对应的转换是 $$[0, 255] \div 255 \times 9 = [0,9]$$，最后用自带的 `round` 进行四舍五入到整数即可。

```python
def grayscale256_to_ascii(val):
    ASCII_MAPPING = "@%#*+=-:. "
    return ASCII_MAPPING[round(val / 255 * (len(ASCII_MAPPING) - 1))]
```

在能将单个像素转换成对应的 ASCII 字符后，接下来我们考虑如何将整个图片（帧）都转换成字符。一张图片本质上可以看成一个二维的 `list`，我们只需要遍历这个 `list`，在每一个像素上都调用之前的 `grayscale256_to_ascii` 函数即可。

```python
from PIL import Image

def image_to_ascii_frame(file):
    img = Image.open(file)
    pixels = img.load()
    return [''.join(grayscale256_to_ascii(pixels[j, i]) \
        for j in range(img.size[0])) for i in range(img.size[1])]
```

这里要注意的就是索引的顺序。一般我们在遍历二维 `list` 的时候，都是先遍历行，再遍历列。`(i, j)` 得到的就是第 `i` 行第 `j` 列的内容。但是这里因为是图片，图片一般先说宽度（列数）再说高度（行数），所以这里为了取得第 `i` 行第 `j` 列的像素，传入的索引实际上是 `(j, i)`。这里使用了第三方的 Pillow 的 `Image` 类来读取图像。

最后我们考虑如何遍历所有的帧，我这里使用的是 Python 自带的 `pathlib` 模组。

```python
from pathlib import Path
from tqdm import tqdm

def images_to_ascii_frames(dir):
    return [image_to_ascii_frame(path) for path in tqdm(sorted(Path(dir).iterdir()))]
```

将 `dir` 这个 `str` 对象转换成 `pathlib.Path` 对象后，因为它对应的是一个目录，所以我们可以通过调用它的 `iterdir` 方法，来得到一个用于遍历这个目录的生成器（Generator）。但是因为这个生成器返回出的图片的顺序不一定按编号，所以我们在外面加上一个 `sorted` 函数来保证这些文件是按照帧的顺序来处理的。`tqdm` 是一个非常神奇的第三方模组，只需要把它套在任何迭代器或者生成器的外面，你就能得到一个非常好看的进度条，简单实用。

在这里两个函数里我都使用了 List Comprehension（中文有人翻译为列表解析式，也有人翻译成列表推导式）。简单来讲 List Comprehension 可以简化列表的构建操作，一行更比四行强（迫真）

```python
# 不使用 List Comprehension
even_numbers = []
for i in range(10):
    if i % 2 == 0:
        even_numbers.append(i)

# 使用 List Comprehension
even_numbers = [i for i in range(10) if i % 2 == 0]
```

至此，视频到 ASCII 字符帧的转换就结束了，接下来我们要考虑如何将这些字符作为动画配上声音播放出来。

## 实现 ASCII 字符动画的播放

虽然说把字符显示到屏幕上只靠自带的 `print` 函数就能完成了，但是为了能让动画的显示效果更好，还是需要费一些心思。

### 让 ASCII 动画的播放与原视频同速

为了能够控制动画播放的速度，我们需要知道原视频的帧率。帧率就是一秒钟有多少帧，它的倒数就是每一帧要持续多长时间。让 ASCII 动画与原视频同速的关键点就是，要让 ASCII 动画每一帧的时长，与原视频中每一帧的时长相同。假设我们的帧率是 2，也就是一秒两帧，每一帧的时长是 0.5 秒，然后我们从第 0 秒开始播放动画，如果 Python 执行代码时所用的时间忽略不计，那么我们在显示完第一帧之后，需要等待 0.5 秒，才能继续显示第二帧。但是我们知道，Python 在执行代码的时候是要花费时间的，而且对于往屏幕上输出字符这种操作来说，是相对耗时的，所以我们并不能忽略掉这些时间，这就导致了我们显示完第一帧之后，需要等待的时间其实是少于 0.5 秒的。那么我们如何判断等待的时间呢？这里可以用下面的代码

```python
import time

frame_rate = 2
frame_len = 1 / frame_rate
start_time = time.time()
for i in range(24):
    print(f'Displaying frame #{i}')
    time.sleep(frame_len - ((time.time() - start_time) % frame_len))
```

其中的 `frame_len` 就是每一帧的时长。

### 播放音频

对于 Windows 10，Python 自带了一个叫 `winsound` 的模组，非常简单易用，支持播放系统声音，播放指定文件的声音，以及播放已经加载到内存里的声音。对于我的 ASCII 动画播放来讲，我需要让它异步播放指定文件。

```python
import winsound as ws
ws.PlaySound('sound.wav', ws.SND_FILENAME | ws.SND_ASNYC)
```

这实际上是调用了 Windows 自带的 Waveform Audio API 里的 [`PlaySound` 函数](https://docs.microsoft.com/en-us/windows/win32/multimedia/the-playsound-function)。

对于 macOS Mojave，虽然 Python 官方没有播放声音的模组，但是 macOS 中自带了一个命令行音乐播放器，名字叫 `afplay`，我们可以使用它来进行音乐播放。

```python
cmd = f'afplay -q 1 "{sound_file}"'
p = sp.Popen(shlex.split(cmd))
try:
    pass  # 用于播放动画的代码（现在暂时省略）
finally:
    p.terminate()
    p.wait()
```

可以看到在 macOS 上播放音乐比在 Windows 上要稍微复杂一点。我们同样是调用了 `sp.Popen` 来执行外部的程序。但是因为我们音乐是要异步播放的，所以我们并没有紧接着调用 `p.wait()`。我将播放动画的逻辑用 `try` 包起来，并在 `finally` 里调用了 `p.terminate()` 和 `p.wait()` 的原因是，因为我们是通过新开进程的方式来执行外部程序，所以在外部程序运行结束后，我们需要“收拾残局”（否则会产生僵尸进程）。此外，我们也希望，无论内部的播放动画的代码出错也好（包括收到了 `KeyboardInterrupt`），运行完成了也好，都应该同时终止音乐的播放。`finally` 中的 `p.terminate()` 和 `p.wait()` 正是用来终止音乐播放并“收拾残局”的两个语句。`p.terminate()` 在执行时会像对应的进程（在这里就是 `afplay`）发送 `SIGTERM` 信号，一般的正常软件在收到这个信号后都会结束自己的运行。如果这个进程已经结束运行（即这个进程已经是僵尸进程）的话，调用 `p.terminate()` 没有任何影响。而紧接着的 `p.wait()` 就是用来“收拾残局”的，在调用时，主进程（这里是 Python）会等待子进程（这里是 `afplay`）退出，然后将它占用的资源彻底释放（就是我说的“收拾残局”）。

至于 `afplay` 如何使用，可以输入 `afplay -h` 查看帮助。

### 使动画顺滑播放

将前两节的代码组合起来，我们可以得到播放 ASCII 动画的代码的原型

```python
import sys
_is_windows = sys.platform == 'win32'
_is_macos = sys.platform == 'darwin'

def play_ascii_frames_with_sound(ascii_frames, frame_rate, sound_file):
    frame_len = 1 / frame_rate
    if _is_windows:
        ws.PlaySound(sound_file, ws.SND_FILENAME | ws.ASYNC)
    elif _is_macos:
        cmd = f'afplay -q 1 "{sound_file}"'
        p = sp.Popen(shlex.split(cmd))
    start_time = time.time()
    try:
        for frame in ascii_frames:
            print(*frame, sep='\n')
            time.sleep(frame_len - ((time.time() - start_time) % frame_len))
    finally:
        if _is_macos:
            p.terminate()
            p.wait()
```

虽然说这段代码能用，但是如果你直接用它进行播放的话，你会发现画面会有上下抖动的情况。这是在显示新一帧之前没有把旧一帧清空导致的。为了清空画面，最先想到的是系统自带的 `clear`（在Windows下是 `cls`）命令。

```python
import sys
_is_windows = sys.platform == 'win32'
_is_macos = sys.platform == 'darwin'

def play_ascii_frames_with_sound(ascii_frames, frame_rate, sound_file):
    frame_len = 1 / frame_rate
    if _is_windows:
        os.system('cls')  # 清空控制台，使动画永远从第一行开始播放，消除抖动
        ws.PlaySound(sound_file, ws.SND_FILENAME | ws.ASYNC)
    else:
        os.system('clear')  # （原因同上）
        if _is_macos:
            cmd = f'afplay -q 1 "{sound_file}"'
            p = sp.Popen(shlex.split(cmd))
    start_time = time.time()
    try:
        for frame in ascii_frames:
            os.system('cls' if _is_windows else 'clear')  # （原因同上）
            print(*frame, sep='\n')
            time.sleep(frame_len - ((time.time() - start_time) % frame_len))
    finally:
        if _is_macos:
            p.terminate()
            p.wait()
```

虽然说这段代码消除了画面上下抖动的问题，但是它又带来了新的问题——闪屏。由于 `clear` 命令会把屏幕清空，所以在新旧两个帧显示的期间，屏幕会有一段时间是空白的。实际运行的时候会发现这样子还是非常破坏观看效果的。那我们有没有办法让新一帧从第一行开始的同时，不要清空屏幕呢？其实解决方法很简单，使用 [ANSI 转义序列（ANSI Escape Code）](https://zh.wikipedia.org/wiki/ANSI%E8%BD%AC%E4%B9%89%E5%BA%8F%E5%88%97)就可以轻易做到这一点了。ANSI 转义序列有很多，但是我们在这里只需要了解用于调整光标位置的序列即可。因为如果我们能直接将光标设置回第一行第一列，我们就能够直接覆盖已有的字符，不需要先清空屏幕了。调整光标位置的代码如下

```python
new_x, new_y = eval(input('Please enter new (x, y): '))
print(f'\x1b[{x};{y}H', end='')
```

其中 `\x1b`（十进制27）代表了 ASCII 码中的第二十七个字符，它是一个控制字符，含义就是转义，用于标志一个转义字符序列的开始。`\x1b[`（这个 `[` 就是普通的方括号）则标志了控制序列导入器（Control Sequence Introducer）的开始。后面的分号 `;` 用于分隔参数，最后的 `H` 可以理解为要执行的操作代码。所以，上面这段代码的作用就是，将光标移动到用户指定的 `(new_x, new_y)` 处。如果省略坐标（即直接输出 `"\x1b[;H"`），那么就默认将光标移动到第一行第一列，这正好是我们想要的。

不过这里头有个小插曲。如果你是Windows用户，你会发现在你输入上述命令之后，光标并没有发生移动，Python 只是直接把这些字符显示了出来而已（因为 `\x1b` 是控制字符，会被显示成一个空格）。造成这种现象的主要原因是，到 2016 年的 Windows 10 1511 版本才开始支持在 Windows 控制台中通过转义序列来更改光标位置（Windows 官方称这项功能为 [Console Virtual Terminal Sequences](https://docs.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences#output-sequences)），并且这项功能并没有被默认启用，启用的方式需要通过系统调用来完成。

在老版本的 Windows 系统中，更改光标位置只能通过系统调用的方式来完成

```cpp
// test.cpp

#include <iostream>
#include <windows.h>

int main(int argc, char** argv) {
    HANDLE stdOutHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (stdOutHandle == INVALID_HANDLE_VALUE) return GetLastError();
    COORD coord = {0, 0};
    if (!SetConsoleCursorPosition(stdOutHandle, coord)) return GetLastError();
    std::cout << "This line should be printed from upper-left corner" << std::endl;
    return 0;
}
```

（你可以通过 `cl test.cpp /EHsc` 来编译这段程序）

在 1511 版本以后的 Windows 10 中，你可以启动 Console Virtual Terminal Sequences 来支持转义序列。

```c++
// test.cpp

#include <iostream>
#include <windows.h>

int main(int argc, char** argv) {
    std::cout << "\x1b[;HNot working..." << std::endl;
    HANDLE outHandle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (outHandle == INVALID_HANDLE_VALUE) return GetLastError();
    DWORD mode = 0;
    if (!GetConsoleMode(outHandle, &mode)) return GetLastError();
    mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if (!SetConsoleMode(outHandle, mode)) return GetLastError();
    std::cout << "\x1b[;HNow it works" << std::endl;
    return 0;
}
```

问题来了，除非你愿意自己给 Python 写一个 C/C++ 扩展，否则我们没有办法在 Python 中调用 Windows 的 API。但是，这里有一个神奇的地方，就是在 Python 里，在使用任何转义序列之前，先用 `os.system` 执行任意命令，在随后的 `print` 函数里你就可以使用转义序列了。

```python
import os

print('\x1b[;HNot working...')  # 此时不能使用转义序列
os.system('echo "whatever command should ok"')
print('\x1b[;H???')  # 奇迹般地就可以用了
```

我查看了 Python 官方关于 [`os.system` 的文档](https://docs.python.org/3/library/os.html#os.system)，发现这个函数存粹是 C 标准库中的 `system` 函数的包装。于是我又去查阅了 MSDN 上关于 [`system` 函数的文档](https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/system-wsystem?view=vs-2019)，里面也没有提到任何关于转义序列的内容，因此我也不知道这背后到底发生了什么。我试着直接使用 C 语言的 `system` 函数，得到的结果跟 Python 中得到的一样。

```c++
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    printf("\x1b[;HNot working...\n");  // 此时不能使用转义序列
    system("echo \"whatever command should ok\"");
    printf("\x1b[;H???\n");  // 奇迹般地就可以用了
    return 0;
}
```

总之，最终得到的播放函数如下

```python
import sys
_is_windows = sys.platform == 'win32'
_is_macos = sys.platform == 'darwin'

def play_ascii_frames_with_sound(ascii_frames, frame_rate, sound_file):
    frame_len = 1 / frame_rate
    if _is_windows:
        os.system('cls')  # 清空控制台，使动画永远从第一行开始播放，消除抖动
        ws.PlaySound(sound_file, ws.SND_FILENAME | ws.ASYNC)
    else:
        os.system('clear')  # （原因同上）
        if _is_macos:
            cmd = f'afplay -q 1 "{sound_file}"'
            p = sp.Popen(shlex.split(cmd))
    start_time = time.time()
    try:
        for frame in ascii_frames:
            print('\x1b[;H', end='')
            print(*frame, sep='\n')
            time.sleep(frame_len - ((time.time() - start_time) % frame_len))
    finally:
        if _is_macos:
            p.terminate()
            p.wait()
```

## 其它的一些工具

至此，整个动画生成和播放的核心功能就完成了。接下来就简单讲一下两个比较重要的工具，一个是 `argparse` 模组，用于处理命令行参数；另一个是 `json` 模组，用于读写 JSON 文件。

### 用 `argparse` 处理命令行参数

首先我们需要创建一个 `argparse.ArgumentParser` 对象，用于处理参数。在创建的时候，我们可以额外加入其它说明文字。

```python
import argparse
parser = argparse.ArgumentParser(epilog='https://github.com/WANGJIEKE/ascii-animation')
```

在上面的代码中，我创建了一个 `ArgumentParser` 对象，并将它的结束语部分设置为这个项目的仓库地址。

往 `parser` 中添加参数也是很容易的。调用它的 `add_argument` 方法即可。

```python
parser.add_argument('-c', '--json-config', help='config JSON file', default='config.json')
parser.add_argument('-o', '--overwrite', help='overwrite file', action='store_true')
```

这里举了两个例子。第一个例子中的 `default` 参数是，如果使用者没有设置这个参数时所使用的默认值。第二个例子中的 `action='store_true'` 参数是，当用户指定了这个参数时，这是这个参数的值为 `True`。

我们可以通过 `parser` 的 `parse_args` 方法来处理参数。它会返回一个包含参数名称和对应的值的对象。

```python
args = parser.parse_args()
if args.json_config:
    print(f'-c (or --json-config) has value "{args.json_config}"')
if args.overwrite:
    print(f'-o (or --overwrite) has value "{args.overwrite}"')
```

### 用 `json` 读写 JSON 文件

```python
import json
with open('file.json', 'w') as f:
    json.dump({"key": "value"}, f)
with open('file.json', 'r') as f:
    json.load(f)
```

JSON 读写在 Python 中是非常轻松的。我在这个项目里使用了 JSON 来保存和读取转换好的 ASCII 字符动画。
