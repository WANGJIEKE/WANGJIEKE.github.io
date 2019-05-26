---
layout: post
title:  "在macOS的shell中快速打开VS Code"
date:   2019-05-25 23:46:00 -0700
tags: study-cs c
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

代码也可以在[我的GitHub Gist](https://gist.github.com/WANGJIEKE/37b7f1d572bfeaa4019e8e5ce258d228)上找到。
