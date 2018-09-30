# NNforMNIST
全连接神经网络，实现MNIST数据集的分类。

####目录结构：
dataloader.py: 数据集读取  
main.py: 程序入口  
network.py: 神经网络定义与实现  
model: 训练得到的模型  
TRAIN_DATA.npy: 训练过程产生的迭代数据(epoch, iteration, accuracy, loss)

####使用方法：
进入main.py运行即可，其中包含三个函数，各负责不同的功能：  
train()用于训练，得到的模型会保存在model文件夹；  
test()用于计算测试集的分类准确率；  
test10RandomImgs()会随机分类10个样本，并使用图形界面显示。
