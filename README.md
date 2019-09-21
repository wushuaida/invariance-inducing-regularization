# Increasing image classification robustness

1.全部代码均使用python 3.7，pytorch1.2

2.spatial transformer network.py 为搭建STN网络对MNIST数据集进行图像分类
model

3.RES18.py 为使用RES18网络对CIFAR数据集进行图像分类

4.myimplement.py和main.py是我融合了RES18网络加入了对抗样本学习方法在CIFAR数据集上进行图像分类，其中myimplement 比较精简，main包含的东西更多。

5.utils.py和models是两个辅助文件，utils是辅助观测训练过程，models提供了很多已经写好的神经网络。