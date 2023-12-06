"""
用于评估模型在数据集上的准确度
"""

import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 确保 CUDA 可用，如果不可用则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'use {device}')

    # 加载预训练的 ResNet 模型
    model = timm.create_model('resnet50', pretrained=True)
    model.to(device)
    model.eval()

    # 定义 ImageNet 数据的预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载 ImageNet 测试集
    imagenet_val = datasets.ImageFolder('ILSVRC2012_img_val_for_ImageFolder', transform=transform)
    imagenet_iter = DataLoader(imagenet_val, batch_size=64, shuffle=False)


    # 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        from tqdm import tqdm
        for i, (images, labels) in tqdm(enumerate(imagenet_iter), desc="calculating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()



    # 打印准确率
    print(f'Accuracy on the ImageNet test set: {100 * correct / total}%')