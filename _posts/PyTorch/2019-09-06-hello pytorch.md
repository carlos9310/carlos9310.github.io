---
layout: post
title: PyTorch笔记
categories: PyTorch
---

PyTorch使用笔记

## 安装
安装时直接参考[官网](https://pytorch.org/get-started/locally/)，官网给出的命令行下载过慢，可通过修改镜像源改善。

初始默认镜像源(-c)为pytorch，相关命令如下：
```
conda install pytorch torchvision cpuonly -c pytorch
```

指定为[清华镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)后的命令如下：
```
conda install pytorch torchvision cpuonly -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```



## 参考
- [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- [https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)
