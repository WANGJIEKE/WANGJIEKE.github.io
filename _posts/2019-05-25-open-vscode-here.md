---
layout: post
title:  "在macOS的shell中快速打开VS Code"
date:   2019-05-25 23:46:00 -0700
tags: study-cs c
---

Visual Studio Code在macOS中是可以通过命令行打开的，具体操作可以在[这篇官方文档](https://code.visualstudio.com/docs/setup/mac#_launching-from-the-command-line)中找到，下面是节选。

> You can also run VS Code from the terminal by typing 'code' after adding it to the path:
>
> - Launch VS Code.
> - Open the **Command Palette** (⇧⌘P) and type 'shell command' to find the **Shell Command: Install 'code' command in PATH** command.
macOS shell commands
>   !['shell command'](/assets/2019-05-25-open-vscode-here/shell-command.png)
> - Restart the terminal for the new `$PATH` value to take effect. You'll be able to type 'code .' in any folder to start editing files in that folder.

以下为毫无卵用的原文章。

---

开始使用Visual Studio Code之后，经常需要在shell中用Visual Studio Code打开某个目录。但是，输入`open . -a "Visual Studio Code"`过于繁琐了，而shell脚本我还没怎么写过，所以我打算用C语言来完成这个脚本。

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define COMMAND_TEMPLATE_LEN 29
// strlen("open  -a \"Visual Studio Code\"") == 29

int main(int argc, char** argv) {
    if (argc > 2) {
        fprintf(stderr, "%s: invalid argument(s)\n", argv[0]);
        fprintf(stderr, "    usage: %s [path]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (argc == 2) {
        char* command = calloc(
            COMMAND_TEMPLATE_LEN + strlen(argv[1]) + 1,
            sizeof(char)
        );
        strcat(command, "open ");
        strcat(command, argv[1]);
        strcat(command, " -a \"Visual Studio Code\"");
        system(command);
        free(command);
    } else {  // argc == 1
        const char* command = "open . -a \"Visual Studio Code\"";
        system(command);
    }
    exit(EXIT_SUCCESS);
}
```

代码完成之后可以macOS自带的`clang`编译。

```bash
cc vscode.c -o vscode
```

最后将二进制文件所在的目录加入`PATH`，重启shell即可。

代码也可以在[我的GitHub Gist](https://gist.github.com/WANGJIEKE/37b7f1d572bfeaa4019e8e5ce258d228)上找到。
