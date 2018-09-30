# dataloader.py：从数据文件中读取样本数据
import os
import struct
import numpy as np

class DataLoader:
    # 初始化函数
    # name: 数据文件名称（'MNIST' / 'CIFAR'）
    # TODO 添加 CIFAR 数据读取
    # path: 数据文件路径
    # mode: 训练数据 or 测试数据
    def __init__(self, name, path):
        self.name = name
        self.path = path

    # 读取数据文件，返回数组形式的样本数据
    # 标签：[N, 1]；图像：[N, 784]，图像大小 [28, 28]
    def Load(self, mode):
        if self.name == 'MNIST':
            train_labels_path = os.path.join(self.path, 'train-labels-idx1-ubyte')
            train_images_path = os.path.join(self.path, 'train-images-idx3-ubyte')

            test_labels_path = os.path.join(self.path, 't10k-labels-idx1-ubyte')
            test_images_path = os.path.join(self.path, 't10k-images-idx3-ubyte')

            # 读取测试数据集 Ground Truth 标签
            with open(train_labels_path, 'rb') as lpath:
                # '>' 表示大端保存（高位存头，低位存尾）
                # 'I' 表示无符号类型
                magic, n = struct.unpack('>II', lpath.read(8))
                train_labels = np.fromfile(lpath, dtype=np.uint8).astype(np.float)

            # 读取测试数据集
            with open(train_images_path, 'rb') as ipath:
                magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
                loaded = np.fromfile(train_images_path, dtype=np.uint8)
                # 图像数据由第 16 个字节开始存储
                train_images = loaded[16:].reshape(len(train_labels), 784).astype(np.float)

            with open(test_labels_path, 'rb') as lpath:
                # '>' 表示大端保存
                # 'I' 表示无符号类型
                magic, n = struct.unpack('>II', lpath.read(8))
                test_labels = np.fromfile(lpath, dtype=np.uint8).astype(np.float)

            with open(test_images_path, 'rb') as ipath:
                magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
                loaded = np.fromfile(test_images_path, dtype=np.uint8)
                # 图像数据由第 16 个字节开始存储
                test_images = loaded[16:].reshape(len(test_labels), 784).astype(np.float)

            if mode == 'train':
                return train_images, train_labels
            else:
                return test_images, test_labels

        else:
            return []