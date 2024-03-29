﻿# Submersible-Search-Model
**潜水器位置预测和搜素救援模型**

该模型用于解决2024数模美赛B题

-----
以下为文件功能说明，具体模型得看论文呢
## 1. terrain.py
将只含有网格状高程数据的数据集制成三维地形图

> 数据集生成方法
> 
> 1. 获取对应地区的DEM文件（例如asc）
>
> 2. 将asc文件的前六行数据参数去除
>
> 3. 将asc文件每行数据前的空格删除（toData.py功能）

## 2. toData.py
去除asc文件每行数据前的空格

## 3. current.py
生成随机有规律的洋流方向和速度

## 4. mist.py  （创新）
利用 *Metropolis-Hastings* 算法根据数据集生成某时间的点在三维空间的分布概率（我们称之为雾状图）

## 5. predict.py    （核心算法1）
首先利用已构建的潜水器4自由度模型，模拟出潜水器的运动轨迹

然后利用MH算法生成潜水器某时刻在海中的概率分布图

## 6. search.py    （核心算法2）
根据已有搜救模型，实现搜救模拟，得到搜救概率与时间函数

---
最后抒发一些感慨：

打数模真的是一件累活，在这4天中真的从早忙到晚，最后一天还得通宵。

不过我觉得还是有必要的。
因为自认为自己比较适合这类比赛，就是在一段时间内专心只搞一件事。

在这段时间内我可以完全投入，沉浸于心流状态，还能学到非常多的知识，至少比那应试教育有意思多了。

参加这比赛也是想要证明自己吧，以前可能真的太低估自己了。加油吧。

希望能获得个好成绩！

