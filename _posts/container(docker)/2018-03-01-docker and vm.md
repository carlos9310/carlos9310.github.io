---
layout: post
title: 虚拟化技术-Docker VS Virtual Machines
categories: docker
---

## 虚拟化
docker是一种**容器化**的虚拟化技术，它与传统的虚拟机(virtual machine)一样，可实现**资源和环境的隔离**。两者实现虚拟化的方式不同，具体对比如下图所示：

![png](/assets/images/vm/docker-vm.png)

传统的虚拟机首先通过**Hypervisor层**对物理硬件进行虚拟化，然后在虚拟的硬件资源上安装**从操作系统(guest os)**，最后将相关**应用**运行在从操作系统上。

而docker不像虚拟机那样利用Hypervisor和guest os实现资源与环境的隔离，其仅通过一个docker daemon/engine来实现**资源限制与环境隔离**(主要利用linux内核本身支持的容器方式来实现这一功能)。 **docker daemon/engine可以简单看成对Linux内核中的NameSpace、Cgroup、镜像管理文件系统操作的封装。** 简单的说，docker利用namespace实现**系统环境的隔离**；利用Cgroup实现**资源限制**；利用镜像实现**根目录环境的隔离**。 

## 小结
由上述分析可知：
- 虚拟机
    - 物理资源层面的隔离(隔离的更彻底，对硬件进行虚拟化，app运行在虚拟的硬件上)
    - 不同虚拟机间系统独立，笨重(G)，启动慢(min)，运行效率低(低于物理硬件)，一台主机可创建有限的虚拟机，资源利用率低
- docker
    - app层面的隔离(隔离性较差，app直接运行在宿主机的内核之上，容器内没有自己的内核，也没进行硬件虚拟化，)
    - 与宿主机共享系统内核，轻快(M)，启动快(s)，运行效率高(接近物理硬件)，一台主机可创建大量容器，资源利用率高
    
说明：
- docker的安全性不高，无法分辨执行指令的用户。A用户可以删除B用户创建的容器，存在一定的安全风险
- docker版本在快速更新中，存在版本兼容问题    
## 参考
* [Docker与虚拟机的区别](https://www.jianshu.com/p/d3006b8a22ee)

* [Docker和虚拟机的对比](https://www.cnblogs.com/jie-fang/p/10279629.html)

* [Docker教程之二Docker和传统虚拟化对比](https://blog.csdn.net/xingfei_work/article/details/81029003)