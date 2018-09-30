# network.py 神经网络定义
# Author: Mingqi Gao, Chongqing University.
# Email: gaomingqi@cqu.edu.cn

import numpy as np

# 计算 softmax 损失函数
def softmax_loss(F3, labels): # (checked)
    """
    :param F3: 网络得到的预测数据，大小[10, batchsize]
    :param labels: 当前 batch 的正确标签集合，大小[1, batchsize]
    :return: 损失值标量，损失函数关于输入数据的梯度，dLoss/dF3 与F3具有相同维度
    """
    shape = np.shape(F3)
    # 计算 softmax 函数：
    # e^(fi) / SUM_i ( e^(fi) )
    exp_F3 = np.exp(F3)
    softmax_F3 = exp_F3 / np.sum(exp_F3, axis=0)

    # 计算 Loss 关于 F3 的损失函数
    dLoss_dF3 = softmax_F3
    # 计算每个样本数据的 Loss
    loss = 0
    for i in range(0, shape[1]):
        # 统计损失
        loss += (-np.log(softmax_F3[labels[i], i]))
        # softmax 梯度计算中，针对目标类别的特殊情况
        dLoss_dF3[labels[i], i] = dLoss_dF3[labels[i], i] - 1

    return loss/shape[1], dLoss_dF3/shape[1]

class NN:
    # 初始化函数
    def __init__(self):
        # 网络结构：X->F1->B1->R1->F2->B2->R2->F3->Out
        # 三个全连接层
        self.F1 = Linear(784, 300)
        self.F2 = Linear(300, 150)
        self.F3 = Linear(150, 10)

        # 两个激活层
        self.R1 = ReLU()
        self.R2 = ReLU()

        # 两个批归一化层
        self.B1 = BN(300)
        self.B2 = BN(150)

        # 记录迭代次数，Loss，验证准确率
        self.TRAIN_DATA = []

    # 网络训练函数
    def train(self, momentum, learning_rate, batchsize, data_loader, valid_rate, epoch, alpha):
        """
        :param momentum: 梯度更新动量
        :param learning_rate: 学习率
        :param batchsize: 批数据量，即同时处理的数据样本数
        :param data_loader: 数据读取接口
        :param valid_rate: 验证数据在训练集中的比例
        :param epoch: 使用完整训练集数据更新参数的次数
        :param alpha: 正则项参数
        :return: 无返回值
        """
        # 记录权重 W1 & W2 &  3的变化过程
        # 初始化权重，采用校准方差方法
        self.W1 = np.random.randn(300, 784) / np.sqrt(784/2)
        self.W2 = np.random.randn(150, 300) / np.sqrt(300/2)
        self.W3 = np.random.randn(10, 150) / np.sqrt(150/2)

        # 初始化 BN 层中使用的权重
        self.gamma1 = np.ones([300, 1], dtype=np.float64)
        self.beta1 = np.zeros([300, 1], dtype=np.float64)
        self.gamma2 = np.ones([150, 1], dtype=np.float64)
        self.beta2 = np.zeros([150, 1], dtype=np.float64)

        # 读入训练数据集、标签
        train_images, train_labels = data_loader.Load('train')
        # labels 类型转换
        train_labels = train_labels.astype(np.uint8)

    # 1. 预处理 (0 均值化)
        mean = np.mean(train_images)
        train_images -= mean
        print('数据集读取结束，训练数据均值: %f' % mean)

        # 数据转置：[784, len(labels)]
        train_images = train_images.T

        # 动量法，速度初始为 0
        v_W1 = 0
        v_W2 = 0
        v_W3 = 0
        v_gamma1 = 0
        v_beta1 = 0
        v_gamma2 = 0
        v_beta2 = 0

    # 开始迭代更新参数
        for e in range(0, epoch):
            for i in range(0, len(train_labels)-len(train_labels)//valid_rate, batchsize):
                # 提取 batchsize 大小的数据子集
                images_batch = train_images[:,i:i+batchsize]
                labels_batch = train_labels[i:i+batchsize]

    # 2. 前向传播
                # 开启、重置梯度记录
                self.F1.setGrad(True)
                self.F2.setGrad(True)
                self.F3.setGrad(True)
                self.R1.setGrad(True)
                self.R2.setGrad(True)

                # 为 W1, W2 设置当前正在更新的权重
                self.F1.setWeight(self.W1)
                self.F2.setWeight(self.W2)
                self.F3.setWeight(self.W3)
                # 为 B1, B2 设置当前正在更新的参数
                self.B1.setParams(self.gamma1, self.beta1)
                self.B2.setParams(self.gamma2, self.beta2)

                # 做前向传播
                F3_out = self.forward(images_batch)

    # 2.2 计算 Loss 函数
                Loss, dLoss_dF3 = softmax_loss(F3_out, labels_batch)

    # 3. 反向传播，计算 相关参数的梯度
                dLoss_dW1, dLoss_dW2, dLoss_dW3, dLoss_dgamma1, dLoss_dbeta1, dLoss_dgamma2, dLoss_dbeta2\
                    = self.backward(dLoss_dF3)

                # 若需要正则项，在损失上增加正则项的值 alpha * R(W) / N
                if alpha > 0:
                    Loss += alpha * (np.sum(np.reshape((self.W1 ** 2).size, 1)) +
                                     np.sum(np.reshape((self.W2 ** 2).size, 1)) +
                                     np.sum(np.reshape((self.W3 ** 2).size, 1))) / len(labels_batch)

                    # 同时计算增加正则项后相关的梯度
                    dLoss_dW1 += alpha * 2 * self.W1 / len(labels_batch)
                    dLoss_dW2 += alpha * 2 * self.W2 / len(labels_batch)
                    dLoss_dW3 += alpha * 2 * self.W3 / len(labels_batch)

    # 4. 更新权重信息
            # 动量法
                v_W1 = momentum * v_W1 - learning_rate * dLoss_dW1
                self.W1 += v_W1
                v_W2 = momentum * v_W2 - learning_rate * dLoss_dW2
                self.W2 += v_W2
                v_W3 = momentum * v_W3 - learning_rate * dLoss_dW3
                self.W3 += v_W3
                v_gamma1 = momentum * v_gamma1 - learning_rate * dLoss_dgamma1
                self.gamma1 += v_gamma1
                v_beta1 = momentum * v_beta1 - learning_rate * dLoss_dbeta1
                self.beta1 += v_beta1
                v_gamma2 = momentum * v_gamma2 - learning_rate * dLoss_dgamma2
                self.gamma2 += v_gamma2
                v_beta2 = momentum * v_beta2 - learning_rate * dLoss_dbeta2
                self.beta2 += v_beta2

                # 每过一段时间，输出一下验证结果
                if (i%100 == 0):
                    acc = self.valid(train_images[:, (len(train_labels)-len(train_labels)//valid_rate):len(train_labels)],
                          train_labels[(len(train_labels)-len(train_labels)//valid_rate):len(train_labels)],
                               batchsize)
                    # 显示当前状态
                    print('epoch-%d, iteration-%d, 当前损失: %f' % (e, i, Loss))

                    self.TRAIN_DATA.append((e, i, acc, Loss))

            # 每 3 个 epoch 就做一次权重衰减
            if e%2 == 0 and epoch > 0:
                learning_rate = learning_rate / 2

        # 训练结束
        print('训练结束')

        # 保存训练好的参数
        np.save('model/W1.npy', self.W1)
        np.save('model/W2.npy', self.W2)
        np.save('model/W3.npy', self.W3)
        np.save('model/gamma1.npy', self.gamma1)
        np.save('model/gamma2.npy', self.gamma2)
        np.save('model/beta1.npy', self.beta1)
        np.save('model/beta2.npy', self.beta2)

        # BN 层 running 参数
        bn1_rm, bn1_rv = self.B1.getRunningParams()
        bn2_rm, bn2_rv = self.B2.getRunningParams()

        np.save('model/bn1_rm.npy', bn1_rm)
        np.save('model/bn2_rm.npy', bn2_rm)
        np.save('model/bn1_rv.npy', bn1_rv)
        np.save('model/bn2_rv.npy', bn2_rv)

        # 保存训练过程的损失值、验证概率信息
        np.save('TRAIN_DATA.npy', np.array(self.TRAIN_DATA))

    # 验证函数，在训练过程中输出验证结果
    def valid(self, images, labels, batchsize, mode='train'):
        """
        :param images: 验证数据集
        :param labels: 验证数据的正确标签集
        :param batchsize: 验证数据批规模
        :return: 返回模型在当前数据集下的分类准确率
        """
        sum = 0
        right_sum = 0
        for i in range(0, len(labels), batchsize):
            out = self.forward(images[:, i:i+batchsize], mode)
            out_gt = labels[i:i+batchsize]

            # 计算 softmax 评分
            # e^(fi) / SUM_i ( e^(fi) )
            exp_out = np.exp(out)
            softmax_out = exp_out / np.sum(exp_out, axis=0)

            # 找到最大值索引，作为分类结果
            idx = np.argmax(softmax_out, axis=0)
            # 与 GT 做对比
            temp = np.array((idx - out_gt)==0, dtype=np.bool)
            # 统计正确分类数量
            right_sum += np.sum(temp)
            sum += batchsize

        sum = float(sum)
        right_sum = float(right_sum)
        acc = right_sum/sum

        print('验证数据集准确率: %f' %acc)

        return acc

    # 前向传播函数
    def forward(self, x, mode = 'train'):
        """
        :param x: 待分类样本数据，大小[data_dim, batchsize]
        :param mode: 'test' or 'train'
        :return: 分类结果，大小[10, batchsize]
        """
        x = self.F1.forward(x)
        x = self.B1.forward(x, mode)
        x = self.R1.forward(x)  # ReLU
        x = self.F2.forward(x)
        x = self.B2.forward(x, mode)
        x = self.R2.forward(x)  # ReLU
        x = self.F3.forward(x)
        # 返回预测结果
        return x

    # 反向传播函数
    def backward(self, top):
        """
        :param top: 上一层神经元梯度
        :return: 待更新的参数梯度，即损失函数关于这些参数的梯度
        """
        # top = dLoss_dF3
        dLoss_dW3, dLoss_dR2 = self.F3.backward(top)

        dLoss_dB2 = self.R2.backward(dLoss_dR2)
        dLoss_dF2, dLoss_dgamma2, dLoss_dbeta2 = self.B2.backward(dLoss_dB2)
        dLoss_dW2, dLoss_dR1 = self.F2.backward(dLoss_dF2)

        dLoss_dB1 = self.R1.backward(dLoss_dR1)
        dLoss_dF1, dLoss_dgamma1, dLoss_dbeta1 = self.B1.backward(dLoss_dB1)
        dLoss_dW1, dLoss_dX = self.F1.backward(dLoss_dF1)

        return dLoss_dW1, dLoss_dW2, dLoss_dW3, dLoss_dgamma1, dLoss_dbeta1, dLoss_dgamma2, dLoss_dbeta2

    # 设置网络参数
    def loadParams(self):
        # 从文件中读取模型参数
        W1 = np.load('model/W1.npy')
        W2 = np.load('model/W2.npy')
        W3 = np.load('model/W3.npy')
        gamma1 = np.load('model/gamma1.npy')
        gamma2 = np.load('model/gamma2.npy')
        beta1 = np.load('model/beta1.npy')
        beta2 = np.load('model/beta2.npy')
        rm1 = np.load('model/bn1_rm.npy')
        rv1 = np.load('model/bn1_rv.npy')
        rm2 = np.load('model/bn2_rm.npy')
        rv2 = np.load('model/bn2_rv.npy')

        # 设置模型参数
        self.F1.setWeight(W1)
        self.F2.setWeight(W2)
        self.F3.setWeight(W3)
        self.B1.setParams(gamma1, beta1)
        self.B1.setRunningParams(rm1, rv1)
        self.B2.setParams(gamma2, beta2)
        self.B2.setRunningParams(rm2, rv2)

# 全连接层定义与实现
class Linear:
    # 初始化函数
    def __init__(self, in_dim, out_dim):
        """
        :param in_dim: 输入数据维度
        :param out_dim: 输出数据维度
        """
        # 使用高斯函数初始化权重分布
        self.W = np.random.randn(out_dim, in_dim) / np.sqrt(in_dim/2)
        # grad 参数用来控制是否需要在前向传播时记录梯度
        self.grad = False
        # 记录全连接层的输入数据，以及权重数据
        self.X = None
        self.WT = None

    # 前向传播函数
    def forward(self, x):
        """
        :param x: 输入数据，大小[in_dim, batchsize]，在实验中，in_dim 为 784, 100, 30
        :return: 分类/映射结果，大小[out_dim, batchsize]，在实验中，out_dim 为 100, 30, 10
        """
        # 计算输出，out = W * X
        out = self.W.dot(x)

        # 保存梯度计算的中间存储数据，只在训练时进行
        if self.grad:
            # 若 D = WX，D 的梯度 dLdD 已知，则 W 的梯度 dLdW 为: dLdD * X^T
            self.X = x.T
            self.WT = self.W.T

        return out

    # 反向传播，计算 dLoss_dX, dLoss_dW
    def backward(self, top):
        """
        :param top: 上一层梯度
        :return: Loss对当前层 W 的梯度，以及对当前层输入 X 的梯度
        """
        # 对权重的梯度
        dW = self.X
        # 对输入的梯度
        dX = self.WT

        return top.dot(dW), dX.dot(top)

    # 改变参数 grad 的值，测试时需要关闭，因为不需要保留中间值来计算梯度
    def setGrad(self, g):
        self.grad = g
        self.X = None
        self.WT = None

    # 将权重 W 设置为当前正在更新的数据
    # 为了在训练过程中持续保持权重最新，其他情况不使用
    def setWeight(self, W):
        self.W = np.array(W, dtype=np.float64)

# BN 层定义与实现
class BN:
    # 初始化函数
    def __init__(self, dim):
        """
        :param dim: 输入数据维度
        """
        # gamma，beta: 调整输入数据范围的参数
        # 让模型自己选择是否再回到原有的分布
        # (in_dim, ) 向量
        self.gamma = np.ones([dim,1])
        self.beta = np.ones([dim,1])

        # 记录训练过程中所有 Batch 的均值与方差之和
        self.running_mean = np.zeros([dim,1], np.float64)
        self.running_var = np.zeros([dim,1], np.float64)

        # 计算梯度的参数
        self.out_ = None
        self.x = None
        self.mean = None
        self.var = None

        self.dim = dim

    # 前向传播
    def forward(self, x, mode):
        """
        :param x: 神经网络的输出, 在本次实验中，大小为：[100, batchsize], [30, batchsize]
        :param mode: 由于 BN 的前向传播过程训练时测试时差别较大，因此在这里需要进行区分
        :return: 返回与 x 具有相同尺寸，但进行了标准高斯化之后的数据
        """
        # 计算每个特征元素的均值 across batches
        # [dim, 1] 矩阵
        out = None

        if mode == 'train':
            # 计算均值, 标准差

            mean_x = np.mean(x, axis=1)
            mean_x = np.reshape(mean_x, [self.dim, 1])

            var_x = np.var(x, axis=1)
            var_x = np.reshape(var_x, [self.dim, 1])

            out_ = (x - mean_x) / np.sqrt(var_x + 1e-3)

            # 更新 running data
            self.running_mean = 0.9 * self.running_mean + (1 - 0.9) * mean_x
            self.running_var = 0.9 * self.running_var + (1 - 0.9) * var_x

            out = self.gamma * out_ + self.beta

            self.out_ = out_
            self.x = x
            self.mean = mean_x
            self.var = var_x

        elif mode == 'test':
            scale = self.gamma / np.sqrt(self.running_var + 1e-3)
            out = x * scale + (self.beta - self.running_mean * scale)

        return out

    # 设置 BN 层的batch衰减均值，测试时要用
    def setRunningParams(self, rm, rv):
        self.running_mean = rm
        self.running_var = rv

    # 获取 BN 层的batch衰减均值，训练结束后需要获取，用来做存储
    def getRunningParams(self):
        return self.running_mean, self.running_var

    # 反向传播
    def backward(self, top):
        """
        :param top: 上一层梯度
        :return: Loss 关于当前层输入的梯度
        """
        N = self.x.shape[0]

        dout_ = self.gamma * top
        dvar = np.sum(dout_ * (self.x - self.mean) * -0.5 * (self.var + 1e-3) ** -1.5, axis=1)
        dvar = dvar[:, np.newaxis]
        dx_ = 1 / np.sqrt(self.var + 1e-3)
        dvar_ = 2 * (self.x - self.mean) / N

        # 用于梯度计算的中间量
        di = dout_ * dx_ + dvar * dvar_
        dmean = -1 * np.sum(di, axis=1)
        dmean_ = np.ones_like(self.x) / N
        dmean = dmean[:, np.newaxis]

        dx = di + dmean * dmean_
        dgamma = np.sum(top * self.out_, axis=1)
        dgamma = dgamma[:, np.newaxis]
        dbeta = np.sum(top, axis=1)
        dbeta = dbeta[:, np.newaxis]

        if len(np.shape(dgamma)) < 2:
            dgamma = dgamma[:, np.newaxis]

        if len(np.shape(dbeta)) < 2:
            dbeta = dbeta[:, np.newaxis]

        return dx, dgamma, dbeta

    # 设置网络参数，仅在训练时使用
    def setParams(self, gamma, beta):
        """
        :param gamma: 更新参数
        :param beta: 更新参数
        :return: 无
        """
        self.gamma = gamma
        self.beta = beta

# 激活层定义与实现（ReLU）
class ReLU:
    # 初始化 ReLU 层
    def __init__(self):
        # grad 参数用来控制是否需要在前向传播时记录梯度
        self.grad = False
        # 记录输入数据
        self.input = None

    # 前向传播函数
    def forward(self, x):
        """
        :param x: 待激活数据，大小[dim, batchsize]，在实验中，dim 为 100, 30
        :return: 与 x 相同大小的激活数据
        """
        out = np.maximum(x, 0)

        if self.grad:
            self.input = x
        # F2 = max(0, F1)
        return out

    # 计算 Loss 对当前层输入的梯度
    def backward(self, top):
        """
        :param top: 上一层梯度
        :return: Loss 对当前层输入的梯度
        """
        dReLU_dX = np.array(self.input, dtype=np.float64)
        dReLU_dX[dReLU_dX > 0] = 1
        dReLU_dX[dReLU_dX < 0] = 0

        return top * dReLU_dX

    # 改变参数 grad 的值，测试时需要关闭，因为不需要保留中间值来计算梯度
    def setGrad(self, g):
        self.grad = g
        self.input = None


# 其他的梯度更新
# RMSProp

                # cache_W1 = 0.9 * cache_W1 + 0.1 * (dLoss_dW1 ** 2)
                # cache_W2 = 0.9 * cache_W2 + 0.1 * (dLoss_dW2 ** 2)
                # cache_W3 = 0.9 * cache_W3 + 0.1 * (dLoss_dW3 ** 2)
                # cache_beta1 = 0.9 * cache_beta1 + 0.1 * (dLoss_dbeta1 ** 2)
                # cache_beta2 = 0.9 * cache_beta2 + 0.1 * (dLoss_dbeta2 ** 2)
                # cache_gamma1 = 0.9 * cache_gamma1 + 0.1 * (dLoss_dgamma1 ** 2)
                # cache_gamma2 = 0.9 * cache_gamma2 + 0.1 * (dLoss_dgamma2 ** 2)
                #
                # self.W1 -= learning_rate * dLoss_dW1 / (np.sqrt(cache_W1) + 1e-7)
                # self.W2 -= learning_rate * dLoss_dW2 / (np.sqrt(cache_W2) + 1e-7)
                # self.W3 -= learning_rate * dLoss_dW3 / (np.sqrt(cache_W3) + 1e-7)
                # self.beta1 -= learning_rate * dLoss_dbeta1 / (np.sqrt(cache_beta1) + 1e-7)
                # self.beta2 -= learning_rate * dLoss_dbeta2 / (np.sqrt(cache_beta2) + 1e-7)
                # self.gamma1 -= learning_rate * dLoss_dgamma1 / (np.sqrt(cache_gamma1) + 1e-7)
                # self.gamma2 -= learning_rate * dLoss_dgamma2 / (np.sqrt(cache_gamma2) + 1e-7)

            # Adam
            #     temp_vector = np.concatenate(
            #         (np.reshape(dLoss_dW1, [dLoss_dW1.shape[0]*dLoss_dW1.shape[1], 1]),
            #         np.reshape(dLoss_dW2, [dLoss_dW2.shape[0]*dLoss_dW2.shape[1], 1]),
            #          np.reshape(dLoss_dW3, [dLoss_dW3.shape[0]*dLoss_dW3.shape[1], 1]),),
            #         axis=0
            #     )
            #
            #     temp_vector = np.concatenate((temp_vector, dLoss_dbeta1, dLoss_dbeta2,
            #                                   dLoss_dgamma1, dLoss_dgamma2), axis=0)
            #
            #     adam_m = beta1 * adam_m + (1-beta1)*temp_vector
            #     adam_v = beta2 * adam_v + (1-beta2)*(temp_vector ** 2)
            #
            #     # adam_m /= 1-beta1**t
            #     # adam_v /= 1-beta2**t
            #
            #     temp_vector = learning_rate * adam_m / (np.sqrt(adam_v)+1e-7)
            #
            #     temp_idx = 0
            #     dlw1 = temp_vector[temp_idx:dLoss_dW1.shape[0] * dLoss_dW1.shape[1]]
            #     temp_idx += dLoss_dW1.shape[0] * dLoss_dW1.shape[1]
            #     dlw2 = temp_vector[temp_idx:temp_idx + dLoss_dW2.shape[0] * dLoss_dW2.shape[1]]
            #     temp_idx += dLoss_dW2.shape[0] * dLoss_dW2.shape[1]
            #     dlw3 = temp_vector[temp_idx:temp_idx + dLoss_dW3.shape[0] * dLoss_dW3.shape[1]]
            #     temp_idx += dLoss_dW3.shape[0] * dLoss_dW3.shape[1]
            #
            #     dlb1 = temp_vector[temp_idx:temp_idx + len(dLoss_dbeta1)]
            #     temp_idx += len(dLoss_dbeta1)
            #     dlb2 = temp_vector[temp_idx:temp_idx + len(dLoss_dbeta2)]
            #     temp_idx += len(dLoss_dbeta2)
            #     dlg1 = temp_vector[temp_idx:temp_idx + len(dLoss_dgamma1)]
            #     temp_idx += len(dLoss_dgamma1)
            #     dlg2 = temp_vector[temp_idx:temp_idx + len(dLoss_dgamma2)]
            #
            #     self.W1 -= np.reshape(dlw1, [self.W1.shape[0], self.W1.shape[1]])
            #     self.W2 -= np.reshape(dlw2, [self.W2.shape[0], self.W2.shape[1]])
            #     self.W3 -= np.reshape(dlw3, [self.W3.shape[0], self.W3.shape[1]])
            #
            #     self.beta1 -= dlb1
            #     self.beta2 -= dlb2
            #     self.gamma1 -= dlg1
            #     self.gamma2 -= dlg2
            #     t += 1

            # 梯度下降SGD
            #     self.W1 -= learning_rate * dLoss_dW1
            #     self.W2 -= learning_rate * dLoss_dW2
            #     self.W3 -= learning_rate * dLoss_dW3
            #     self.beta1 -= learning_rate * dLoss_dbeta1
            #     self.beta2 -= learning_rate * dLoss_dbeta2
            #     self.gamma1 -= learning_rate * dLoss_dgamma1
            #     self.gamma2 -= learning_rate * dLoss_dgamma2
