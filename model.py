# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    model.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了CustomNet类，用于定义神经网络模型
# -----------------------------------------------------------------------

import torch
from torch import nn

class CustomNet(nn.Module):
    """自定义神经网络模型。
    请完成对__init__、forward方法的实现，以完成CustomNet类的定义。
    """

    def __init__(self, input_size=64*64, hidden_size=64*64, num_layers=2, num_classes=10):
        """初始化方法。
        在本方法中，请完成神经网络的各个模块/层的定义。
        请确保每层的输出维度与下一层的输入维度匹配。
        """
        super(CustomNet, self).__init__()

        # START----------------------------------------------------------
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # END------------------------------------------------------------

    def forward(self, x):
        """前向传播过程。
        在本方法中，请完成对神经网络前向传播计算的定义。
        """
        # START----------------------------------------------------------
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        x = x.view(x.size(0), -1, self.hidden_size)  # Flatten the image tensor
        out, _ = self.rnn(x, h0)  
        out = self.fc(out[:, -1, :])
        return out
        # END------------------------------------------------------------


if __name__ == "__main__":
    # 测试
    from dataset import CustomDataset
    from torchvision.transforms import ToTensor

    c = CustomDataset('./images/train.txt', './images/train', ToTensor)
    net = CustomNet()                                # 实例化
    x = torch.unsqueeze(c[10]['image'], 0)      # 模拟一个模型的输入数据
    print(net.forward(x))                            # 测试forward方法
