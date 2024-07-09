# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    inference.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了用于在模型应用端进行推理，返回模型输出的流程
# -----------------------------------------------------------------------

import torch
from PIL import Image
from torchvision.transforms import ToTensor


def inference(image_path, model, device):
    """定义模型推理应用的流程。
    :param image_path: 输入图片的路径
    :param model: 训练好的模型
    :param device: 模型推理使用的设备，即使用哪一块CPU、GPU进行模型推理
    """
    # 将模型置为评估（测试）模式
    model.eval()

    # START----------------------------------------------------------
    # 加载图像
    image = Image.open(image_path)
    # 将图像转换为张量
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    # 进行推理
    with torch.no_grad():
        output = model(image_tensor)
    # 获取预测结果
    _, predicted = torch.max(output.data, 1)
    print(f"Predicted class: {predicted.item()}")
    # END------------------------------------------------------------


if __name__ == "__main__":
    # 指定图片路径
    image_path = "./images/test/signs/img_0006.png"

    # 加载训练好的模型
    model = torch.load('./models/model.pkl')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 显示图片，输出预测结果
    inference(image_path, model, device)
