import torch
import torch.nn.functional as F
import torch.nn as nn
from config import *
from typing import Tuple, List
class SequentialWithInternalStatePrediction(nn.Sequential):
    """
    实现了predict_internal_states功能的Sequential的适应版本
    """

    def predict_internal_states(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        将子模块应用于输入并返回所有中间输出。
        参数:
            x (torch.Tensor): 输入张量。
        返回:
            Tuple[List[torch.Tensor], torch.Tensor]: 包含每个层的中间输出的列表和最终输出的元组。
        """
        result = []
        for module in self:
            x = module(x)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                result.append(x)
        return result, x

class MLP_teacher(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(MLP_teacher, self).__init__()#三层全连接网络
        self.fc1 = nn.Linear(784, 1200)  # 输入层-隐藏层,784像素映射到1200个神经元
        self.fc2 = nn.Linear(1200, 120)  # 1200个神经元映射成1200个神经元
        self.fc3 = nn.Linear(120, num_classes)   # 1200个神经元映射成10个类别
        self.relu = nn.ReLU()           # ReLU激活函数
        self.dropout = nn.Dropout(p=0.5)#增加Dropout层,防止过拟合

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入转换为批次大小 x 784 的形状
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x =self.fc3(x)
        return x
    
class MLP_student(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(MLP_student, self).__init__()#三层全连接网络
        self.fc1 = nn.Linear(784, 120)  # 输入层-隐藏层,784像素映射到1200个神经元
        self.fc2 = nn.Linear(120, 12)  # 1200个神经元映射成1200个神经元
        self.fc3 = nn.Linear(12, num_classes)   # 1200个神经元映射成10个类别
        self.relu = nn.ReLU()           # ReLU激活函数
        self.dropout = nn.Dropout(p=0.5)#增加Dropout层,防止过拟合

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入转换为批次大小 x 784 的形状
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x =self.fc3(x)
        return x
    
""" class MNISTModel_teacher(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTModel_teacher, self).__init__()
        # 定义了模型的特征提取部分
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 30,  kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(30*6*6, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.9)#增加Dropout层,防止过拟合
    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))  # Feature extraction stage 1
        x = self.pool(F.relu(self.conv2(x)))  # Feature extraction stage 2
        x = x.view(-1, 30*6*6)            # Flatten the tensor
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.softmax(self.dropout(self.fc2(x)), dim=1)
        # x = self.fc2(self.dropout(x))
        return x """
class MNISTModel_teacher(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTModel_teacher, self).__init__()
        # 定义了模型的特征提取部分
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.9)  # 增加Dropout层,防止过拟合
    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))  # Feature extraction stage 1
        x = self.pool(F.relu(self.conv2(x)))  # Feature extraction stage 2
        x = x.view(-1, 16 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.dropout(self.fc1(x)))
        return self.fc2(x)
    
class Net_MNIST_student(nn.Module):
    def __init__(self, num_classes=10):
        super(Net_MNIST_student, self).__init__()
        # 定义了模型的特征提取部分
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 12)
        self.fc2 = nn.Linear(12, num_classes)
        self.dropout = nn.Dropout(p=0.9)  # 增加Dropout层,防止过拟合

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))  # Feature extraction stage 1
        x = self.pool(F.relu(self.conv2(x)))  # Feature extraction stage 2
        x = x.view(-1, 16 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.dropout(self.fc1(x)))
        # x = F.relu(self.fc1(x))
        return self.fc2(x)         
    
class Net_CIFAR10_teacher(nn.Module):
    # Alexnet
    def __init__(self):
        super(Net_CIFAR10_teacher, self).__init__()
        # Define the layers of the network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256*1*1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        self.dropout = nn.Dropout(p=0.2)  # 增加Dropout层,防止过拟合
    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv3(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv4(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv5(x)))  # 16x16 -> 8x8
        x = x.view(-1, 256 * 1 * 1)           # Flatten the tensor
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x

class Net_CIFAR10_student(nn.Module):
    # Alexnet_student
    def __init__(self):
        super(Net_CIFAR10_student, self).__init__()
        # Define the layers of the network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256*1*1, 409)
        self.fc2 = nn.Linear(409, 40)
        self.fc3 = nn.Linear(40, 10)
        self.dropout = nn.Dropout(p=0.2)  # 增加Dropout层,防止过拟合
    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv3(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv4(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv5(x)))  # 16x16 -> 8x8
        x = x.view(-1, 256 * 1 * 1)           # Flatten the tensor
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    

class Resblk(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Resblk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual Layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Classification Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)  # 10 classes for CIFAR-10

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(Resblk(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Resblk(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

""" class Net_MNIST_teacher(nn.Module):
    # LeNet-5
    def __init__(self):#适用CIFAR10图像分类任务的典型CNN，两个卷积层和三个全连接层
        super(Net_MNIST_teacher, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)#创建一个卷积层，输入通道数为1，输出通道数为6，卷积核大小为5x5。
        self.pool = nn.AvgPool2d(2, 2)#创建一个平均池化层，池化窗口大小为2x2。
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)#创建另一个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5。
        self.sigmoid = nn.Sigmoid()#Sigmoid激活函数
        self.flatten = nn.Flatten()#展平层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#创建一个全连接层，输入大小为16x5x5，输出大小为120。
        self.fc2 = nn.Linear(120, 84)#创建另一个全连接层，输入大小为120，输出大小为84。
        self.fc3 = nn.Linear(84, 10)#创建最后一个全连接层，输入大小为120，输出大小为84。

    def forward(self, x):
        x = self.pool(self.sigmoid(self.conv1(x)))#将输入x通过卷积层self.conv1，然后通过Sigmoid激活函数，再通过池化层self.pool进行处理。
        x = self.pool(self.sigmoid(self.conv2(x)))#将处理后的结果再次通过卷积层self.conv2，然后通过Sigmoid激活函数，再通过池化层self.pool进行处理。
        # x = x.view(-1, 16 * 5 * 5)#将处理后的结果展平为一个向量,卷积核大小为5*5
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))#然后通过全连接层self.fc1，
        x = self.sigmoid(self.fc2(x))#再次通过全连接层self.fc2，
        return self.fc3(x)#最后通过全连接层self.fc3 """

""" class Net_MNIST(nn.Module):
    def __init__(self):
        super(Net_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 输入层-隐藏层
        self.fc2 = nn.Linear(128, 10)   # 隐藏层-输出层
        self.relu = nn.ReLU()           # ReLU激活函数

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入转换为批次大小 x 784 的形状
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x    
 """

""" class Net_CIFAR10(nn.Module):#定义网络模型架构
    def __init__(self):#适用CIFAR10图像分类任务的典型CNN，两个卷积层和三个全连接层
        super(Net_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#创建一个卷积层，输入通道数为3，输出通道数为6，卷积核大小为5x5。
        self.pool = nn.MaxPool2d(2, 2)#创建一个最大池化层，池化窗口大小为2x2。
        self.conv2 = nn.Conv2d(6, 16, 5)#创建另一个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5。
        self.fc1 = nn.Linear(16 * 5 * 5, 1200)#创建一个全连接层，输入大小为16x5x5，输出大小为1200。
        self.fc2 = nn.Linear(1200, 100)#创建另一个全连接层，输入大小为1200，输出大小为100。
        self.fc3 = nn.Linear(100, 10)#创建最后一个全连接层，输入大小为100，输出大小为10。

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#将输入x通过卷积层self.conv1，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = self.pool(F.relu(self.conv2(x)))#将处理后的结果再次通过卷积层self.conv2，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = x.view(-1, 16 * 5 * 5)#将处理后的结果展平为一个向量,卷积核大小为5*5
        x = F.relu(self.fc1(x))#然后通过全连接层self.fc1，再通过ReLU激活函数。
        x = F.relu(self.fc2(x))#再次通过全连接层self.fc2，再通过ReLU激活函数。
        return self.fc3(x)#最后通过全连接层self.fc3 """

""" class Net_CIFAR10_student(nn.Module):#定义网络模型架构
    def __init__(self):#适用CIFAR10图像分类任务的典型CNN，两个卷积层和三个全连接层
        super(Net_CIFAR10_student, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#创建一个卷积层，输入通道数为3，输出通道数为6，卷积核大小为5x5。
        self.pool = nn.MaxPool2d(2, 2)#创建一个最大池化层，池化窗口大小为2x2。
        self.conv2 = nn.Conv2d(6, 16, 5)#创建另一个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5。
        self.fc1 = nn.Linear(16 * 5 * 5, 20)#创建一个全连接层，输入大小为16x5x5，输出大小为20。
        self.fc2 = nn.Linear(20, 20)#创建另一个全连接层，输入大小为20，输出大小为20。
        self.fc3 = nn.Linear(20, 10)#创建最后一个全连接层，输入大小为20，输出大小为10。

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#将输入x通过卷积层self.conv1，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = self.pool(F.relu(self.conv2(x)))#将处理后的结果再次通过卷积层self.conv2，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = x.view(-1, 16 * 5 * 5)#将处理后的结果展平为一个向量,卷积核大小为5*5
        x = F.relu(self.fc1(x))#然后通过全连接层self.fc1，再通过ReLU激活函数。
        x = F.relu(self.fc2(x))#然后通过全连接层self.fc2，再通过ReLU激活函数。
        return self.fc3(x)#最后通过全连接层self.fc3 """

def load_model(model_name="Net_CIFAR10"):
    if model_name == "Net_MNIST":
        if is_MLP == True and dataset_name != "CIFAR10":
            return MLP_teacher().to(DEVICE)
        else:
            return MNISTModel_teacher().to(DEVICE)  # 返回模型并转换到正确的设备
    if model_name == "Net_CIFAR10":
        return Net_CIFAR10_teacher().to(DEVICE)  # 返回模型并转换到正确的设备
    if model_name == "ResNet18":
        return ResNet18().to(DEVICE)
    if model_name == "Net_MNIST_student":
        if is_MLP == True and dataset_name != "CIFAR10":
            return MLP_student().to(DEVICE)
        else:
            return Net_MNIST_student().to(DEVICE)
    if model_name == "Net_CIFAR10_student":
        return Net_CIFAR10_student().to(DEVICE)  # 返回模型并转换到正确的设备
        