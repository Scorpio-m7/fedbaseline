from dataset import *
import torch
import copy
from torch.utils.data import DataLoader, Subset
if torch.backends.mps.is_available() :
    DEVICE = torch.device("mps")#mac调用gpu训练

def local_train(net, trainloader, epochs):#根据训练集和训练次数训练网络
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    for _ in range(epochs):#循环训练次数
        for images,labels in trainloader:
            optimizer.zero_grad()#梯度清零
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()#将图像数据送入模型并转换至设备，计算模型输出与真实标签之间的交叉熵损失。然后反向传播计算参数梯度。
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

def average_weights(global_model, local_weights):#计算全局模型的平均权重
    global_dict = global_model.state_dict()#获取全局模型的参数字典
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_weights[i][key].float() for i in range(len(local_weights))], 0).mean(0)#计算平均权重
    global_model.load_state_dict(global_dict)#加载平均权重

def federated_learning_fedavg(global_model, trainset, num_clients, epochs_per_round, num_rounds, noniid=False, device=DEVICE):
    client_data = create_clients(trainset, num_clients, noniid)
    for round in range(num_rounds):
        local_weights = []
        for client in range(num_clients):
            local_model = copy.deepcopy(global_model)
            client_train_data = DataLoader(Subset(trainset, client_data[client]), batch_size=32, shuffle=True)
            local_train(local_model, client_train_data, epochs_per_round)
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            print(f'Client {client + 1}/{num_clients} trained')
        average_weights(global_model, local_weights)
        print(f'Round {round + 1}/{num_rounds} completed')
    return global_model

def federated_learning_fedprox(global_model, trainset, num_clients, epochs_per_round, num_rounds, mu=0.01, noniid=False, device=DEVICE):
    client_data = create_clients(trainset, num_clients, noniid)
    for round in range(num_rounds):
        local_weights = []
        global_weights = list(global_model.parameters())
        for client in range(num_clients):
            local_model = copy.deepcopy(global_model)
            client_train_data = DataLoader(Subset(trainset, client_data[client]), batch_size=32, shuffle=True)
            fedprox_local_train(local_model, global_weights, client_train_data, epochs_per_round, mu)
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            print(f'Client {client + 1}/{num_clients} trained')
        average_weights(global_model, local_weights)
        print(f'Round {round + 1}/{num_rounds} completed')
    return global_model

def federated_learning_scaffold(global_model, trainset, num_clients, epochs_per_round, num_rounds, lr=0.001, noniid=False, device=DEVICE):
    client_data = create_clients(trainset, num_clients, noniid)
    # 列表推导式遍历 `global_model.parameters()`，获取模型的所有参数。对于每个参数 `param`，调用 `torch.zeros_like(param)` 创建一个与 `param` 形状和数据类型相同但全为零的新张量，并添加到 `c_global` 列表中
    c_global = [torch.zeros_like(param) for param in global_model.parameters()]
    # 外层循环 `num_clients` 次。对于每个客户端，内层列表推导式创建一个与 `global_model` 参数形状和类型相匹配的零张量列表。这些列表作为内层列表推导式的结果，被添加到外层列表 `c_locals` 中，形成一个二维列表结构。
    c_locals = [[torch.zeros_like(param) for param in global_model.parameters()] for _ in range(num_clients)]
    for round in range(num_rounds):
        local_weights = []
        # 遍历`global_model.parameters()`来创建与全局模型参数形状相同的零张量，以存储每个客户端对控制变量（`c_global`）的更新量
        delta_c_locals = [[torch.zeros_like(param) for param in global_model.parameters()] for _ in range(num_clients)]
        for client in range(num_clients):
            local_model = copy.deepcopy(global_model)
            client_train_data = DataLoader(Subset(trainset, client_data[client]), batch_size=32, shuffle=True)
            scaffold_local_train(local_model, client_train_data, epochs_per_round, c_locals[client], c_global, lr)
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            print(f'Client {client + 1}/{num_clients} trained')
            
            with torch.no_grad():
                # 在无梯度模式下，计算每个客户端控制变量的更新量，即将本地控制变量减去全局控制变量，结果存储在`delta_c_locals`中
                for delta_c_param, local_c_param, global_c_param in zip(delta_c_locals[client], c_locals[client], c_global):
                    delta_c_param.data = local_c_param - global_c_param
        average_weights(global_model, local_weights)#全局模型参数更新
        with torch.no_grad():
            for c_global_param,delta_c_local in zip(c_global, zip(*delta_c_locals)):
                # 无梯度模式，遍历`c_global`和`delta_c_locals`的转置（即所有客户端的参数更新量），计算新的全局控制变量
                c_global_param.data += (1 / num_clients) * sum(delta_c for delta_c in delta_c_local)
                # 每个全局控制变量的更新是通过取所有客户端对应参数更新量的平均值并加到当前全局控制变量上实现的。
        print(f'Round {round + 1}/{num_rounds} completed')
    return global_model