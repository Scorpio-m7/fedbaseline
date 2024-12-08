from dataset import *
import torch
import copy
from torch.utils.data import DataLoader, Subset
from client import *
from config import *
from datetime import datetime

def save_model_values_to_file(model, input_data, round_num):
    # 确保输入数据和模型都在同一个设备上
    input_data = input_data.to(device)
    model = model.to(device)
    # 计算每轮聚合后的数值
    with torch.no_grad():
        x = input_data.view(-1, 784)  # 将输入转换为批次大小 x 784 的形状
        hidden = model.relu(model.fc1(x))
        output = model.fc2(hidden)
    # 保存数值到文件
    with open(f"model_values_round_{round_num}.txt", "w") as f:
        f.write("Input Layer:\n")
        for i in range(784):
            f.write(f"Neuron {i}: {input_data.view(-1, 784)[0, i]:.2f}\n")
        f.write("\nHidden Layer:\n")
        for i in range(128):
            f.write(f"Neuron {i}: {hidden[0, i]:.2f}\n")
        f.write("\nOutput Layer:\n")
        for i in range(10):
            f.write(f"Neuron {i}: {output[0, i]:.2f}\n")
def test(net, testloader):#评估函数，并计算损失和准确率    
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    correct,total, loss = 0, 0,0.0#初始化正确分类的数量、总样本数量、损失值
    with torch.no_grad():#禁用梯度计算
        for images,labels in testloader:
            images = images.to(DEVICE)  # 确保输入图像在正确的设备上
            labels = labels.to(DEVICE)  # 确保标签被转换为 Tensor 并在正确的设备上
            outputs=net(images)#图像传给模型
            if malicious_ratio>0:
                for i in range(labels.size(0)):
                    if labels[i] != 7:#除去中毒的测试数据集
                        loss += criterion(outputs[i].unsqueeze(0), labels[i].unsqueeze(0)).item()  # 累计模型损失
                        total+=1
                        correct += (torch.max(outputs[i].unsqueeze(0).data, 1)[1] == labels[i].unsqueeze(0)).sum().item()  # 累加正确数量
            else:
                for i in range(labels.size(0)):
                    loss += criterion(outputs[i].unsqueeze(0), labels[i].unsqueeze(0)).item()  # 累计模型损失
                    total += 1
                    correct += (torch.max(outputs[i].unsqueeze(0).data, 1)[1] == labels[i].unsqueeze(0)).sum().item()  # 累加正确数量
    return round(loss/len(testloader.dataset),4), round(correct/total,4)#返回损失和准确度

def plot_results(losses, accuracies, asrs, dataset_name,malicious_ratio,noniid,model_exchange,num_rounds,attack_type):    
    if not osp.exists('plt'):
        os.makedirs('plt')
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
    max_accuracy = max(accuracies)
    max_accuracy_round = accuracies.index(max_accuracy)
    plt.annotate(# 标记 Accuracy 的峰值位置
        f'Max Accuracy: {max_accuracy*100:.2f}%\nRound {max_accuracy_round}',
        xy=(max_accuracy_round, max_accuracy),
        xytext=(max_accuracy_round + num_rounds * 0.1, max_accuracy - 0.1),
        arrowprops=dict(facecolor='green', arrowstyle='->'),
        fontsize=10,
        color='green'
    )
    last_asr_value = asrs[-1]# 获取最后一个值
    last_asr_round = asrs.index(last_asr_value)# 获取最后一个值的索引
    max_asr_value = max(asrs) 
    max_asr_round = asrs.index(max_asr_value)
    plt.annotate( # 标记 ASR 最后一轮的位置
        f"last ASR: {last_asr_value*100:.2f}%\nRound: {last_asr_round}",
        xy=(last_asr_round, last_asr_value),
        xytext=(last_asr_round + 5, last_asr_value + 0.05),
        arrowprops=dict(facecolor='red', arrowstyle='->'),
        fontsize=10,
        color='red'
    )
    plt.annotate( # 标记 ASR 的最低位置
        f"peak ASR: {max_asr_value*100:.2f}%\nRound: {max_asr_round}",
        xy=(max_asr_round, max_asr_value),
        xytext=(max_asr_round + 5, max_asr_value + 0.05),
        arrowprops=dict(facecolor='blue', arrowstyle='->'),
        fontsize=10,
        color='blue'
    )
    plt.xlabel('Round')
    plt.ylabel('Accuracy / ASR')
    plt.title('Accuracy and ASR over Rounds')
    plt.legend()
    plt.tight_layout()
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'plt/{dataset_name}_{noniid}_{current_time}_{malicious_ratio}_{model_exchange}_{num_rounds}_{attack_type}.png')

def ASR(net, testloader, target_label):#评估后门样本，并计算ASR
    """# 展示前32张图片和标签
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images = images[:32].to(DEVICE)  # 取前32张图片
    labels = labels[:32].to(DEVICE)  # 取前32个标签
    # 可视化前32张图片和标签
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        image = images[i].cpu().numpy().transpose((1, 2, 0))  # 转换为 (H, W, C) 格式
        label = labels[i].item()
        ax.imshow(image)
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.show() """
    if malicious_ratio<=0:
        return 0
    correct, total = 0, 0
    with torch.no_grad():#禁用梯度计算
        for images, labels in testloader:
            images = images.to(DEVICE)  # 确保输入图像在正确的设备上
            labels = labels.to(DEVICE)  # 确保标签也在正确的设备上            
            outputs = net(images)#图像传给模型
            predicted = torch.max(outputs.data, 1)[1]#预测输出
            # print(f"predicted: {predicted}")                   
            for i in range(labels.size(0)):
                if labels[i] == 7:
                    if predicted[i] == target_label:
                        correct += 1
                    total += 1
    return round(correct/total,4) if total > 0 else 0 # 返回ASR
def average_weights(global_model, local_weights):#计算全局模型的平均权重
    global_dict = global_model.state_dict()#获取全局模型的参数字典
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_weights[i][key].float() for i in range(len(local_weights))], 0).mean(0)#计算平均权重
    global_model.load_state_dict(global_dict)#加载平均权重

def sort_neurons_by_activation(model, data_loader,dataset_name, layer_name):
    model.eval()
    activations = []
    step = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            if dataset_name == 'MNIST':
                if layer_name == 'fc1':
                    outputs = model.fc1(images.view(-1, 784))
                elif layer_name == 'fc2':
                    x = model.fc1(images.view(-1, 784))
                    x = model.relu(x)
                    outputs = model.fc2(x)
                elif layer_name == 'fc3':
                    x = model.fc1(images.view(-1, 784))
                    x = model.relu(x)
                    x = model.fc2(x)
                    x = model.relu(x)
                    outputs = model.fc3(x)    
            if dataset_name == 'CIFAR10':
                # 对于 CIFAR10，经过卷积层、池化、再获取 fc1 的激活输出
                x = F.relu(model.conv1(images))    # conv1 -> ReLU
                x = model.pool(x)                  # 池化
                x = F.relu(model.conv2(x))         # conv2 -> ReLU
                x = model.pool(x)                  # 池化
                x = x.view(-1, 16 * 5 * 5)         # 展开为全连接层输入大小
                if layer_name == 'fc1':
                    outputs = model.fc1(x)             # 获取 fc1 层激活输出                
                elif layer_name == 'fc2':
                    x = model.fc1(x)
                    x = F.relu(x)
                    outputs = model.fc2(x)
                elif layer_name == 'fc3':
                    x = model.fc1(x)
                    x = F.relu(x)
                    x = model.fc2(x)
                    x = F.relu(x)
                    outputs = model.fc3(x)
            activations.append(outputs.cpu().numpy())
            step += 1
    activations = np.vstack(activations)   # 计算平均激活值并排序
    mean_activations = np.mean(activations, axis=0)
    # sorted_indices = np.argsort(-mean_activations)  # 从大到小排序，返回索引
    # sorted_indices = np.argsort(mean_activations)  # 从小到大排序，返回索引
    sorted_indices = np.arange(mean_activations.shape[0])# 不进行排序，直接返回索引
    return sorted_indices
def replace_neurons(local_model, student_model, sorted_indices, layer_name,num_neurons=20,mix_ratio=0.7):
    with torch.no_grad():
        if layer_name == 'fc1':
            local_weights = local_model.fc1.weight.data
            local_biases = local_model.fc1.bias.data
            student_weights = student_model.fc1.weight.data
            student_biases = student_model.fc1.bias.data
        elif layer_name == 'fc2':
            local_weights = local_model.fc2.weight.data
            local_biases = local_model.fc2.bias.data
            student_weights = student_model.fc2.weight.data
            student_biases = student_model.fc2.bias.data
        elif layer_name == 'fc3':
            local_weights = local_model.fc3.weight.data
            local_biases = local_model.fc3.bias.data
            student_weights = student_model.fc3.weight.data
            student_biases = student_model.fc3.bias.data
        # 计算 Kronecker 积
        kron_weights = torch.kron(student_weights, torch.ones(local_weights.shape[0] // student_weights.shape[0], local_weights.shape[1] // student_weights.shape[1]).to(DEVICE))
        kron_biases = torch.kron(student_biases, torch.ones(local_biases.shape[0] // student_biases.shape[0]).to(DEVICE))
        for i in range(min(num_neurons, len(sorted_indices))):# 替换最激活的 num_neurons 个神经元
            index = sorted_indices[i]
            if index < local_weights.shape[0]:
                local_weights[index] = mix_ratio * kron_weights[i] + (1 - mix_ratio) * local_weights[index]# 更新权重,按照混合比例进行混合
                local_biases[index] = mix_ratio * kron_biases[i] + (1 - mix_ratio) * local_biases[index]
def fedavg(global_model, student_model,trainset,testset ,dataset_name,num_clients,epochs_per_round, num_rounds, target_label,malicious_ratio,noniid=False):
    model_exchange=False
    if malicious_ratio > 0:
        if dataset_name == 'MNIST':
            _, testset_malicious = load_malicious_data_mnist(attack_type)
        elif dataset_name == 'CIFAR10':
            _, testset_malicious = load_malicious_data_CIFAR10(attack_type)
    testset_malicious=testset
    losses = []
    accuracies = []
    asrs = []
    for round in range(num_rounds):
        local_weights = []
        for client in range(num_clients):
            local_model = copy.deepcopy(global_model)
            # if client < num_clients*malicious_ratio and round >= start_malicious_round and round < end_malicious_round:
            if client < num_clients*malicious_ratio:
                if dataset_name == 'MNIST':
                    malicious_trainloader, _ = load_malicious_data_mnist(attack_type)
                elif dataset_name == 'CIFAR10':
                    malicious_trainloader, _ = load_malicious_data_CIFAR10(attack_type)
                malicious_trainset=malicious_trainloader.dataset
                client_data_malicious = create_clients(malicious_trainset, num_clients, noniid)
                # 如果是恶意数据集，则使用client_data_malicious函数加载恶意数据
                client_train_data_malicious = DataLoader(Subset(malicious_trainset, client_data_malicious[client]), batch_size=64, shuffle=True)#创建客户端训练集
                local_train(local_model, student_model,client_train_data_malicious, epochs_per_round, client_id=client, round_num=round)
                print(f'malicious Client {client + 1}/{num_clients} trained in round {round + 1}')
            else:      
                clear_trainset=trainset          
                client_data = create_clients(clear_trainset, num_clients, noniid)#创建客户端数据集
                client_train_data = DataLoader(Subset(clear_trainset, client_data[client]), batch_size=64, shuffle=True)#创建客户端训练集
                local_train(local_model, student_model,client_train_data, epochs_per_round, client_id=client, round_num=round)
                if (client+1) % 10 == 0:
                    print(f'Client {client + 1}/{num_clients} trained in round {round + 1}')
            # if client < num_clients*malicious_ratio and round >= start_malicious_round and round < end_malicious_round:
            #     sorted_indices = sort_neurons_by_activation(local_model, client_train_data,dataset_name, 'fc1')
            #     replace_neurons(local_model, student_model, sorted_indices,'fc1')# Replace the first 20 neurons in local_model with those from student_model
            #     sorted_indices = sort_neurons_by_activation(local_model, client_train_data,dataset_name, 'fc2')
            #     replace_neurons(local_model, student_model, sorted_indices,'fc2')# Replace the first 20 neurons in local_model with those from student_model
            #     sorted_indices = sort_neurons_by_activation(local_model, client_train_data,dataset_name, 'fc3')
            #     replace_neurons(local_model, student_model, sorted_indices,'fc3')# Replace the first 20 neurons in local_model with those from student_model
            #     model_exchange=True
            local_weights.append(copy.deepcopy(local_model.state_dict()))
        average_weights(global_model, local_weights)
        """ input_data = torch.randn(1, 784)  # 这里假设使用随机输入数据，实际应用中可以使用真实数据
        save_model_values_to_file(global_model, input_data, round, device) """
        loss, accuracy = test(global_model, testset)
        losses.append(loss)
        accuracies.append(accuracy) 
        asr = ASR(global_model, testset_malicious, target_label)        
        asrs.append(asr)
        # print(f'Round {round + 1}/{num_rounds} completed')
    plot_results(losses, accuracies, asrs,dataset_name,malicious_ratio,noniid,model_exchange,num_rounds,attack_type)
    return global_model

def fedprox_loss(local_model, global_model, mu):
    prox_loss = 0.0
    for param_local, param_global in zip(local_model.parameters(), global_model.parameters()):
        prox_loss += (mu / 2) * torch.norm(param_local - param_global) ** 2
    return prox_loss

def fedprox(global_model, trainset, num_clients, epochs_per_round, num_rounds, mu=0.01, noniid=False):
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

def scaffold(global_model, trainset, num_clients, epochs_per_round, num_rounds, lr=0.001, noniid=False):
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

