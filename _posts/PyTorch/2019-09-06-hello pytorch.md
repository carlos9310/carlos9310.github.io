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


## gather(聚集)/scatter_(分散)
- 聚集操作 torch.gather(input, dim, index, out=None) → Tensor

    沿着某个轴(dim)方向，按照输入(input)的索引张量(index)中指定的位置从input中聚集形成一个新张量(out)，且**out与index的形状相同**。对一个3维张量，输出可定义为：
    ```
    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2 
    ```
    具体例子为：
    ```
    b = torch.Tensor([[1,2,3],[4,5,6]])
    print(b) 
    index_1 = torch.LongTensor([[0,1],[2,0]])
    print(torch.gather(b, dim=1, index=index_1))
    # out[i][j] = b[i][index_1[i][j]]
    
    index_2 = torch.LongTensor([[0,1,1],[0,0,0]])
    print(torch.gather(b, dim=0, index=index_2)) 
    # out[i][j] = b[index_1[i][j]][j]
    ```
    输出：
    ```
    tensor([[1., 2., 3.],
        [4., 5., 6.]])
    tensor([[1., 2.],
            [6., 4.]])
    tensor([[1., 5., 6.],
            [1., 2., 3.]])
    ```

- 分散操作 input.scatter_(dim, index, src) → Tensor (input为某个Tensor)
    
    按照index张量中指定的位置将src张量值分散到指定的input张量中。对于一个三维张量，input更新为(src中无对应索引位置时，input对应位置的元素保持不变)：
    ```
        input[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        input[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        input[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
    ```
    具体例子
    ```
    input1 = torch.zeros(3, 5)
    src = torch.rand(2, 5)
    print(src)
    # input1[index[i][j]][j]= src[i][j]
    print(input1.scatter_(0, torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]]), src))
    input2 = torch.zeros(2, 4)
    # input2[i][index[i][j]]= 1.23
    print(input2.scatter_(1, torch.tensor([[2], [3]]), 1.23))
    ```
    输出
    ```
    tensor([[0.8834, 0.5526, 0.6427, 0.4812, 0.5709],
        [0.9993, 0.9984, 0.0662, 0.2923, 0.0377]])
    tensor([[0.8834, 0.9984, 0.0662, 0.4812, 0.5709],
            [0.0000, 0.5526, 0.0000, 0.2923, 0.0000],
            [0.9993, 0.0000, 0.6427, 0.0000, 0.0377]])
    tensor([[0.0000, 0.0000, 1.2300, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.2300]])
    ```
    
 


## 参考
- [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- [https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)
- [Pytorch 学习（5）：Pytorch中的 torch.gather/scatter_ 聚集/分散操作](https://blog.csdn.net/duan_zhihua/article/details/82556676)
