
# Increasing image classification robustness

1.本项目为吴达文的毕业论文《基于对抗样本学习的提升图像分类鲁棒性研究》相关代码实现

2.全部代码均使用python 3.7，pytorch1.2

3.spatial transformer network.py 为搭建STN网络对MNIST数据集进行图像分类
model

4.RES18.py 为使用RES18网络对CIFAR数据集进行图像分类

5.myimplement.py和main.py是我融合了RES18网络加入了对抗样本学习方法在CIFAR数据集上进行图像分类，其中myimplement 比较精简，main包含的东西更多。

6.utils.py和models是两个辅助文件，utils是辅助观测训练过程，models提供了很多已经写好的神经网络。

------
2019.9.21

目前已完成：

1.普通的RES18网络对CIFAR10数据集进行图像分类

2.加入对抗样本学习方法进入RES18神经网络对CIFAR10数据集进行图像分类

3.使用STN网络对MNIST数据集进行图像分类




下一步工作：

1.调整选取对抗样本的随即方法

2.将STN的图像转换部件加入到我的模型中








