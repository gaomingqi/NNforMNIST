from dataloader import DataLoader
from network import NN
import numpy as np
import matplotlib.pyplot as plt
import time

# 基于Python的 3-层全连接神经网络，实现MNIST数据集分类.
# Author: Mingqi Gao, Chongqing University.
# Email: gaomingqi@cqu.edu.cn

# 显示随机10个样本的分类结果
def test10RandomImgs():
    # 读取测试集结果
    dl = DataLoader('MNIST', './datasets/MNIST/')
    test_images, test_labels = dl.Load('test')
    # 生成10个连续的随机数
    idx = np.random.randint(0, test_images.shape[0] - 10)
    random_image = test_images[idx:idx + 10, :]
    # testlabel = nn.test(random_image.T)

    MNIST_test = NN()
    MNIST_test.loadParams()
    out = MNIST_test.forward(random_image.T-33.318421, 'test')

    exp_out = np.exp(out)
    softmax_out = exp_out / np.sum(exp_out, axis=0)
    # 找到最大值索引，作为分类结果
    cls_res = np.argmax(softmax_out, axis=0)

    # 显示图像
    for i in range(0, 10):
        plt.subplot(2, 5, i+1)
        plt.title('分类: %d' % cls_res[i])
        # 关闭坐标刻度
        plt.xticks([])
        plt.yticks([])
        plt.imshow(np.reshape(random_image[i], [28, 28]), cmap='gray')

    plt.show()

def train():
    # 读取训练数据集
    dl = DataLoader('MNIST', './datasets/MNIST/')
    train_images, train_labels = dl.Load('train')

    # 预处理 (0 均值化)
    mean = np.mean(train_images)
    print('数据集读取结束，训练数据均值: %f' % mean)
    train_images -= mean
    print('训练数据集数量: %d' % len(train_labels))

    # 训练
    startTime = time.time()
    # 初始化网络
    MNIST_Net = NN()
    MNIST_Net.train(momentum=0.9, learning_rate=0.01, batchsize=100, data_loader=dl, valid_rate=10, epoch=10, alpha=0)
    endTime = time.time()
    print('训练时间：%d' % (endTime - startTime))

def test(mean=33.318421):
    # 读取测试数据集，注意：测试时的均值仍然要与训练时均值相同
    dl = DataLoader('MNIST', './datasets/MNIST/')
    test_images, test_labels = dl.Load('test')

    # 预处理 (0 均值化)
    test_images -= mean
    print('测试数据集数量: %d' % len(test_labels))

    # 测试
    MNIST_Net = NN()
    MNIST_Net.loadParams()
    MNIST_Net.valid(test_images.T, test_labels, 100, mode = 'test')

# -----------------------------------------------------
# MAIN，程序入口
# -----------------------------------------------------

# 训练网络
# train()

# 测试网络
# test()

# 随机分类10个样本，并使用窗口显示，若没有看到窗口显示，请查看是否在编译器窗口的下面
# 我在Pycharm上遇到过这个问题，正在找原因，但结果是正确的
test10RandomImgs()