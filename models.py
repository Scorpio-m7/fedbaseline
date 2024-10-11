import torch
import torch.nn.functional as F
import torch.nn as nn
from config import *

class Net_CIFAR10(nn.Module):#定义网络模型架构
    def __init__(self):#适用CIFAR10图像分类任务的典型CNN，两个卷积层和三个全连接层
        super(Net_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#创建一个卷积层，输入通道数为3，输出通道数为6，卷积核大小为5x5。
        self.pool = nn.MaxPool2d(2, 2)#创建一个最大池化层，池化窗口大小为2x2。
        self.conv2 = nn.Conv2d(6, 16, 5)#创建另一个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#创建一个全连接层，输入大小为16x5x5，输出大小为120。
        self.fc2 = nn.Linear(120, 84)#创建另一个全连接层，输入大小为120，输出大小为84。
        self.fc3 = nn.Linear(84, 10)#创建最后一个全连接层，输入大小为84，输出大小为10。

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#将输入x通过卷积层self.conv1，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = self.pool(F.relu(self.conv2(x)))#将处理后的结果再次通过卷积层self.conv2，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = x.view(-1, 16 * 5 * 5)#将处理后的结果展平为一个向量,卷积核大小为5*5
        x = F.relu(self.fc1(x))#然后通过全连接层self.fc1，再通过ReLU激活函数。
        x = F.relu(self.fc2(x))#再次通过全连接层self.fc2，再通过ReLU激活函数。
        return self.fc3(x)#最后通过全连接层self.fc3

class Net_enhanced_CIFAR10(nn.Module):
    def __init__(self):
        super(Net_enhanced_CIFAR10, self).__init__()
        # Define the layers of the network
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Use adaptive pooling here
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv3(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv4(x)))  # 16x16 -> 8x8
        x = self.adaptive_pool(x)             # Apply adaptive pooling
        x = x.view(-1, 512 * 8 * 8)           # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_MNIST(nn.Module):
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

class Net_MNIST_teacher(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(Net_MNIST_teacher, self).__init__()#三层全连接网络
        self.fc1 = nn.Linear(784, 1200)  # 输入层-隐藏层,784像素映射到1200个神经元
        self.fc2 = nn.Linear(1200, 1200)  # 1200个神经元映射成1200个神经元
        self.fc3 = nn.Linear(1200, num_classes)   # 1200个神经元映射成10个类别
        self.relu = nn.ReLU()           # ReLU激活函数
        self.dropout = nn.Dropout(p=0.5)#增加Dropout层,防止过拟合

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入转换为批次大小 x 784 的形状
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x =self.fc3(x)
        return x
    
class Net_MNIST_student(nn.Module):
    def __init__(self,in_channels=1,num_classes=10):
        super(Net_MNIST_student, self).__init__()#三层全连接网络
        self.fc1 = nn.Linear(784, 20)  # 输入层-隐藏层,784像素映射到20个神经元
        #self.fc2 = nn.Linear(20, 20)  # 20个神经元映射成20个神经元
        self.fc3 = nn.Linear(20, num_classes)   # 20个神经元映射成10个类别
        self.relu = nn.ReLU()           # ReLU激活函数

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入转换为批次大小 x 784 的形状
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x =self.fc3(x)
        return x

class Net_CIFAR10_student(nn.Module):#定义网络模型架构
    def __init__(self):#适用CIFAR10图像分类任务的典型CNN，两个卷积层和三个全连接层
        super(Net_CIFAR10_student, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#创建一个卷积层，输入通道数为3，输出通道数为6，卷积核大小为5x5。
        self.pool = nn.MaxPool2d(2, 2)#创建一个最大池化层，池化窗口大小为2x2。
        self.conv2 = nn.Conv2d(6, 16, 5)#创建另一个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5。
        self.fc1 = nn.Linear(16 * 5 * 5, 20)#创建一个全连接层，输入大小为16x5x5，输出大小为20。
        self.fc3 = nn.Linear(20, 10)#创建最后一个全连接层，输入大小为20，输出大小为10。

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#将输入x通过卷积层self.conv1，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = self.pool(F.relu(self.conv2(x)))#将处理后的结果再次通过卷积层self.conv2，然后通过ReLU激活函数，再通过池化层self.pool进行处理。
        x = x.view(-1, 16 * 5 * 5)#将处理后的结果展平为一个向量,卷积核大小为5*5
        x = F.relu(self.fc1(x))#然后通过全连接层self.fc1，再通过ReLU激活函数。
        return self.fc3(x)#最后通过全连接层self.fc3

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

def load_model(model_name="Net_CIFAR10"):
    if model_name == "Net_MNIST":
        return Net_MNIST().to(DEVICE)  # 返回模型并转换到正确的设备
    if model_name == "Net_CIFAR10":
        return Net_CIFAR10().to(DEVICE)  # 返回模型并转换到正确的设备
    if model_name == "ResNet18":
        return ResNet18().to(DEVICE)
    if model_name == "Net_enhanced_CIFAR10":
        return Net_enhanced_CIFAR10().to(DEVICE)
    if model_name == "Net_MNIST_student":
        return Net_MNIST_student().to(DEVICE)
    if model_name == "Net_CIFAR10_student":
        return Net_CIFAR10_student().to(DEVICE)  # 返回模型并转换到正确的设备
    
