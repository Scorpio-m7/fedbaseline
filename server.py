from dataset import *
import torch
import copy
from torch.utils.data import DataLoader, Subset
from client import *
from config import *

def test(net, testloader):#评估函数，并计算损失和准确率    
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    correct,total, loss = 0, 0,0.0#初始化正确分类的数量、总样本数量、损失值
    with torch.no_grad():#禁用梯度计算
        for images,labels in testloader:
            images = images.to(DEVICE)  # 确保输入图像在正确的设备上
            labels = labels.to(DEVICE)  # 确保标签被转换为 Tensor 并在正确的设备上
            outputs=net(images)#图像传给模型
            loss += criterion(outputs, labels).item()#累计模型损失
            total+=labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()#累加正确数量
    return loss/len(testloader.dataset),correct/total#返回损失和准确度

def plot_results(losses, accuracies, asrs, dataset_name,noniid):    
    # 绘制准确率和损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_rounds), losses, label='Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Loss over Rounds')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(num_rounds), accuracies, label='Accuracy')
    plt.plot(range(num_rounds), asrs, label='ASR', linestyle='--')
    plt.xlabel('Round')
    plt.ylabel('Accuracy / ASR')
    plt.title('Accuracy and ASR over Rounds')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_with_{noniid}_baseline.png')
def ASR(net, testloader, target_label):#评估后门样本，并计算ASR
    correct, total = 0, 0
    with torch.no_grad():#禁用梯度计算
        for images, labels in testloader:
            images = images.to(DEVICE)  # 确保输入图像在正确的设备上
            labels = labels.to(DEVICE)  # 确保标签也在正确的设备上
            outputs = net(images)#图像传给模型
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == target_label).sum().item()#累加正确数量
    return correct/total  # 返回ASR
def average_weights(global_model, local_weights):#计算全局模型的平均权重
    global_dict = global_model.state_dict()#获取全局模型的参数字典
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_weights[i][key].float() for i in range(len(local_weights))], 0).mean(0)#计算平均权重
    global_model.load_state_dict(global_dict)#加载平均权重

def federated_learning_fedavg(global_model, trainset,testset ,dataset_name,num_clients,epochs_per_round, num_rounds, malicious_ratio=0,noniid=False, device=DEVICE,start_malicious_round=1):
    losses = []
    accuracies = []
    asrs = []
    for round in range(num_rounds):
        local_weights = []
        for client in range(num_clients):
            local_model = copy.deepcopy(global_model)
            if client < num_clients*malicious_ratio and round >= start_malicious_round:
                #如果trainset是mnist数据集，则使用load_malicious_data_mnist函数加载恶意数据
                if isinstance(trainset, MNIST):
                    trainloader_mnist, _ = load_malicious_data_mnist()
                    trainset=trainloader_mnist.dataset
                    client_data = create_clients(trainset, num_clients, noniid)
                elif isinstance(trainset, CIFAR10):
                    trainloader_cifar10, _ = load_malicious_data_CIFAR10()
                    trainset=trainloader_cifar10.dataset
                    client_data = create_clients(trainset, num_clients, noniid)
                print(f'malicious Client {client + 1}/{num_clients} trained in round {round + 1}')
            else:
                if (client+1) % 5 == 0:
                    print(f'Client {client + 1}/{num_clients} trained in round {round + 1}')
            client_data = create_clients(trainset, num_clients, noniid)#创建客户端数据集
            client_train_data = DataLoader(Subset(trainset, client_data[client]), batch_size=32, shuffle=True)#创建客户端训练集
            local_train(local_model, client_train_data, epochs_per_round)
            local_weights.append(copy.deepcopy(local_model.state_dict()))
        average_weights(global_model, local_weights)
        loss, accuracy = test(global_model, testset)
        losses.append(loss)
        accuracies.append(accuracy)
        asr = ASR(global_model, testset, target_label)
        asrs.append(asr)
        if round % 5 == 0:
            print(f'Round {round + 1}/{num_rounds} completed')
    plot_results(losses, accuracies, asrs,dataset_name,noniid)
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

