import torch
from config import *

def local_train(net, trainloader, epochs):#根据训练集和训练次数训练网络
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    for _ in range(epochs):#循环训练次数
        for images,labels in trainloader:
            images, labels = net(images.to(DEVICE)), labels.to(DEVICE)
            optimizer.zero_grad()#梯度清零
            criterion(images, labels).backward()#将图像数据送入模型并转换至设备，计算模型输出与真实标签之间的交叉熵损失。然后反向传播计算参数梯度。
            optimizer.step()#梯度更新

def fedprox_local_train(net, global_weights, trainloader, epochs, mu=0.01):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))            
            proximal_term = 0.0
            #遍历本地模型net的所有参数以及全局模型参数global_weights。对于每一对本地参数和全局参数，计算它们之间的L2范数（即欧几里得距离），并将其累加到proximal_term中。这一步骤的目的是为了确保本地模型在更新时不会偏离全局模型太远。
            for param, global_param in zip(net.parameters(), global_weights):
                proximal_term += (param - global_param.to(DEVICE)).norm(2)
            #将计算出的FedProx正则化项乘以超参数mu后的一半，添加到原始的交叉熵损失上，形成最终的损失函数。mu是调节正则化强度的参数，较大的mu值意味着更强的正则化效果。
            loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            
def scaffold_local_train(net, trainloader, epochs, c_i, c_global, lr=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()#将图像数据送入模型并转换至设备，计算模型输出与真实标签之间的交叉熵损失。然后反向传播计算参数梯度。
            optimizer.step()
            for param, c_i_param, c_global_param in zip(net.parameters(), c_i, c_global):
                param.grad.data += c_i_param - c_global_param
            # 所有Tensor的操作都不会被记录到计算图中，即不会计算梯度。这对于不需要梯度的计算（如模型推理或参数更新）来说，可以显著减少内存消耗并提高计算速度。
            with torch.no_grad():
                #  `net.parameters()`包含了神经网络的所有可学习参数；`c_i`和`c_global`分别代表了局部参数和全局参数。
                for param, c_i_param, c_global_param in zip(net.parameters(), c_i, c_global):
                    # 减去学习率乘以（梯度加上局部参数与全局参数的差值），实现了对局部参数`c_i_param`的更新
                    c_i_param.data -= lr * (param.grad + c_i_param - c_global_param)
            
            # Simulate Malicious Behavior
