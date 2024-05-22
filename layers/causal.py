# -*-coding: utf-8 -*-
# @Time    : 2023/1/6 21:19
# @Author  : Feng Li
# @File    : causal.py
# @Software: PyCharm
# 导入库
"""
参考 TCN模型   https://blog.csdn.net/qq_34107425/article/details/105522916
"""

import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import torch.fft
import torch
from scipy import fft, signal


class FFTFilter:
    def __init__(self, fs=16000, fpass=(0.5, 250), filt_order=8):
        self.fs = fs
        self.fpass = fpass
        self.filt_order = filt_order

    def fit_transform(self, x):
        # FFT
        X = torch.fft.fft(x)  # 对 seq_len 这个维度进行 FFT
        freqs = fft.fftfreq(x.shape[-2], d=1 / self.fs)  # 计算频率轴

        # 滤波器设计（带通滤波，保留 fpass[0]~fpass[1] 的成分）
        nyquist = self.fs / 2  # 奈奎斯特频率
        bp_filt_b, bp_filt_a = signal.butter(self.filt_order, (self.fpass[0] / nyquist, self.fpass[1] / nyquist),
                                             btype='bandpass')  # 巴特沃斯带通滤波器

        # 实施滤波
        for i in range(x.shape[0]):  # 对 batch_size 进行循环
            x[i] = torch.tensor(signal.filtfilt(bp_filt_b, bp_filt_a, x[i].cpu().detach().numpy(), axis=-2,
                                                method='gust').copy())  # 使用FFT加速滤波

        return torch.abs(X), x


class FourierFilter(nn.Module):
    def __init__(self, freq_cutoff, sampling_rate, filter_type='lowpass'):
        super(FourierFilter, self).__init__()
        self.freq_cutoff = freq_cutoff
        self.sampling_rate = sampling_rate
        self.filter_type = filter_type

    def forward(self, x):
        batch_size, num_channels, seq_len = x.size()
        x = x.cpu().detach().numpy()
        x_fft = np.apply_along_axis(lambda x: np.fft.fft(x), axis=2, arr=x)
        freqs = np.fft.fftfreq(seq_len, 1.0 / self.sampling_rate)
        if self.filter_type == 'lowpass':
            mask = np.abs(freqs) < self.freq_cutoff
        elif self.filter_type == 'highpass':
            mask = np.abs(freqs) > self.freq_cutoff
        else:
            raise ValueError("Invalid filter type: {}".format(self.filter_type))
        x_fft[:, :, mask] = 0
        x_filtered = np.real(np.fft.ifft(x_fft))
        x_filtered = torch.from_numpy(x_filtered).float()
        return x_filtered.view(batch_size, num_channels, seq_len).cuda()


# 卷积结束后会因为padding导致卷积之后的新数据的尺寸B>输入数据的尺寸A，所以只保留输出数据中前面A个数据；
# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Chomp1d(nn.Module):
    """
    可以看出来，这个函数就是第一个数据到倒数第chomp_size的数据，这个chomp_size就是padding的值。比方说输入数据是5，padding是1，那么会产生6个数据没错吧，那么就是保留前5个数字。
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out, kernel_size=3, stride=1, dilation=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        self.res_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        res_x = self.res_conv(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + res_x


# 这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.05):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数或者特征数
        :param n_outputs: int, 输出通道数或者特征数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长, 在TCN固定为1
        :param dilation: int, 膨胀系数. 与这个Residual block(或者说, 隐藏层)所在的层数有关系.
                                例如, 如果这个Residual block在第1层, dilation = 2**0 = 1;
                                      如果这个Residual block在第2层, dilation = 2**1 = 2;
                                      如果这个Residual block在第3层, dilation = 2**2 = 4;
                                      如果这个Residual block在第4层, dilation = 2**3 = 8 ......
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        # 因为 padding 的时候, 在序列的左边和右边都有填充, 所以要裁剪
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 1×1的卷积. 只有在进入Residual block的通道数与出Residual block的通道数不一样时使用.
        # 一般都会不一样, 除非num_channels这个里面的数, 与num_inputs相等. 例如[5,5,5], 并且num_inputs也是5
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # 在整个Residual block中有非线性的激活. 这个容易忽略!
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.05):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数或者特征数
        :param num_channels: list
        每层的hidden_channel数. 例如[5,12,3], 代表有3个block,
                                block1的输出channel数量为5;
                                block2的输出channel数量为12;
                                block3的输出channel数量为3.
        每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # 可见，如果num_channels=[5,12,3]，那么
        # block1的dilation_size=1
        # block2的dilation_size=2
        # block3的dilation_size=4
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, out_size):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], out_size)
        # self.fe = FourierFilter(freq_cutoff=10, sampling_rate=20)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.fe(x)
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        out = self.linear(output)
        out = self.dropout(out)
        return out


if __name__ == '__main__':
    # for example
    # 输入27个通道，或者特征
    x = torch.randn(2, 32, 27)
    # 构建1层的TCN，最后输出一个通道，或者特征
    model2 = TemporalConvNet(num_inputs=27, num_channels=[32, 16, 10, 1], kernel_size=3, dropout=0.3)
    out = model2(x)
    print(out.shape)
