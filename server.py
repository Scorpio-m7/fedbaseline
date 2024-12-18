from dataset import *
from client import *
from config import *
from useless import *
import cv2
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA

def test(net, testloader):#评估函数，并计算损失和准确率    
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    correct,total, loss = 0, 0,0.0#初始化正确分类的数量、总样本数量、损失值
    with torch.no_grad():#禁用梯度计算
        for images,labels in testloader:
            images = images.to(DEVICE)  # 确保输入图像在正确的设备上
            labels = labels.to(DEVICE)  # 确保标签被转换为 Tensor 并在正确的设备上
            outputs=net(images)#图像传给模型
            # 排除标签为7的样本
            if malicious_ratio > 0:
                mask = labels != 7  # 创建掩码，去掉标签为7的样本
                images, labels = images[mask], labels[mask]
                outputs = outputs[mask]  # 更新输出，仅保留非7标签样本
            # 计算损失（批量处理）
            loss += criterion(outputs, labels).item()*images.size(0)
            # 计算准确率
            _, predicted = torch.max(outputs, 1)  # 获取预测标签
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            """ if malicious_ratio>0:
                for i in range(labels.size(0)):
                    if labels[i] != 7:#除去中毒的测试数据集
                        loss += criterion(outputs[i].unsqueeze(0), labels[i].unsqueeze(0)).item()  # 累计模型损失
                        total+=1
                        correct += (torch.max(outputs[i].unsqueeze(0).data, 1)[1] == labels[i].unsqueeze(0)).sum().item()  # 累加正确数量
            else:
                for i in range(labels.size(0)):
                    loss += criterion(outputs[i].unsqueeze(0), labels[i].unsqueeze(0)).item()  # 累计模型损失
                    total += 1
                    correct += (torch.max(outputs[i].unsqueeze(0).data, 1)[1] == labels[i].unsqueeze(0)).sum().item()  # 累加正确数量 """
    print("Loss:", round(loss/total,4), "Accuracy:", round(correct/total,4))
    return round(loss/total,4), round(correct/total,4)#返回损失和准确度

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
    save_path =f'plt/{dataset_name}_{noniid}_{current_time}_{malicious_ratio}_{model_exchange}_{num_rounds}_{attack_type}.png'
    plt.savefig(save_path)
    if not osp.exists('logger'):
        os.makedirs('logger')
    log_folder = f'./logger/{dataset_name}'
    os.makedirs(log_folder, exist_ok=True)
    log_filename = os.path.join(log_folder, f"training_log_{dataset_name}.log")
    logging.basicConfig(filename=log_filename,level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"finish training with {num_rounds} rounds, {num_clients} clients, {epochs_per_round} epochs per round, {noniid} noniid, {malicious_ratio} malicious_ratio, {dataset_name} dataset, {model_exchange} model_exchange, {start_malicious_round} start_malicious_round, {end_malicious_round} end_malicious_round, {max_asr_value} max_asr_value ,{max_accuracy} max_accuracy , {attack_type} attack_type {current_time}")
    logging.info(f"Saved plot to {save_path}")
    logging_file(f'server.py')
    logging_file(f'config.py')
    logging_file(f'client.py')

def logging_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    logging.info(f"the {file_path} is :\n{content}")
def ASR(net, testloader, target_label):#评估后门样本，并计算ASR
    """ # 展示前32张图片和标签
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
    print(f"ASR: {round(correct/total,4)}")
    return round(correct/total,4) if total > 0 else 0 # 返回ASR
def average_weights(global_model, local_weights):#计算全局模型的平均权重
    global_dict = global_model.state_dict()#获取全局模型的参数字典
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_weights[i][key].float() for i in range(len(local_weights))], 0).mean(0)#计算平均权重
    global_model.load_state_dict(global_dict)#加载平均权重
def aggregate_soft_labels(local_soft_labels, local_true_labels, global_model, testset):
    # Calculate performance of each client
    client_accuracies = []
    for soft_labels, true_labels in zip(local_soft_labels, local_true_labels):
        predictions = np.argmax(soft_labels, axis=1)
        accuracy = np.mean(predictions == true_labels)
        client_accuracies.append(accuracy)

    # Normalize accuracies to get weights
    total_accuracy = sum(client_accuracies)
    client_weights = [acc / total_accuracy for acc in client_accuracies]

    # Aggregate soft labels using weighted averaging
    num_classes = global_model.fc3.out_features if hasattr(global_model, 'fc3') else global_model.fc.out_features
    aggregated_soft_labels = np.zeros((len(local_soft_labels[0]), num_classes))

    for soft_labels, weight in zip(local_soft_labels, client_weights):
        aggregated_soft_labels += weight * soft_labels

    return aggregated_soft_labels
def sort_neurons_by_activation(model, data_loader,dataset_name, layer_name,round):
    model.eval()
    activations = []
    step = 0
    with torch.no_grad():
        for images, label in data_loader:
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
    # sorted_indices = np.arange(mean_activations.shape[0])# 不进行排序，直接返回索引
    # sorted_indices = np.argsort(-mean_activations)  # 从大到小排序，返回索引
    sorted_indices = np.argsort(mean_activations) # 从小到大排序，返回索引
    """ # 计算注意力图
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images = images.to(DEVICE)  # 取前32张图片
    labels = labels.to(DEVICE)  # 取前32个标签
    attention_maps = np.mean(activations, axis=1)
    attention_maps = [normalize_attention_map(amap) for amap in attention_maps]
    attention_maps = [amap.reshape(-1, 1) if amap.ndim == 1 else amap for amap in attention_maps]
    # 可视化注意力图
    visualize_attention_maps(images,labels, attention_maps, layer_name,round) """
    return sorted_indices
def replace_neurons(local_model, student_model, sorted_indices, layer_name,mix_ratio,eta):
    with torch.no_grad():
        local_weights = getattr(local_model, layer_name).weight.data
        local_biases = getattr(local_model, layer_name).bias.data
        student_weights = getattr(student_model, layer_name).weight.data
        student_biases = getattr(student_model, layer_name).bias.data
        # # 计算 Kronecker 积
        # student_weights = torch.kron(student_weights, torch.ones(local_weights.shape[0] // student_weights.shape[0], local_weights.shape[1] // student_weights.shape[1]).to(DEVICE))    
        # student_biases = torch.kron(student_biases, torch.ones(local_biases.shape[0] // student_biases.shape[0]).to(DEVICE))

        """ # 使用零填充将学生模型的权重和偏置扩大到本地模型的大小
        pad_height = local_weights.shape[0] - student_weights.shape[0]# 获取本地模型和学生模型的形状
        pad_width = local_weights.shape[1] - student_weights.shape[1]# 计算需要填充的大小
        student_weights = torch.nn.functional.pad(student_weights, (0, pad_width, 0, pad_height), "constant", 0)
        student_biases = torch.nn.functional.pad(student_biases.unsqueeze(0), (0, pad_height), "constant", 0).squeeze(0) """
        
        # 使用L2 范数来确定优先替换的权重区域
        importance = torch.norm(local_weights, p=2, dim=1)  # 按行计算 L2 范数
        sorted_indices = torch.argsort(importance, descending=True)
        
        for i in range(min(student_weights.shape[0], len(sorted_indices))):# 替换最激活的 2 个神经元
        # for i in range(int(1*len(sorted_indices))):# 替换20%激活的神经元
            index = sorted_indices[i]
            if index < local_weights.shape[0]:
                student_weight = student_weights[i]  # 使用L2范数
                # padded_student_weight = torch.cat([student_weight, torch.zeros(local_weights.shape[1] - student_weight.shape[0]).to(student_weights.device)], dim=0)# 使用0填充
                padded_student_weight = torch.cat([student_weight, local_weights[index, student_weight.shape[0]:]], dim=0)# 使用本地权重填充
                local_weights[index] = eta*(mix_ratio  *padded_student_weight + (1 - mix_ratio) * local_weights[index])# 更新权重,按照混合比例进行混合
                # local_weights[index] = eta*(mix_ratio  * student_weights[i] + (1 - mix_ratio) * local_weights[index])# 更新权重,按照混合比例进行混合
                local_biases[index] = eta*(mix_ratio * student_biases[i] + (1 - mix_ratio) * local_biases[index])#神经元放大eta倍
def fedavg(global_model, student_model,trainset,testset ,dataset_name,num_clients,epochs_per_round, num_rounds, target_label,malicious_ratio,noniid=False):
    testset_malicious=testset
    clear_trainset=trainset 
    if malicious_ratio > 0:
        if dataset_name == 'MNIST':
            malicious_trainloader, testset_malicious = load_malicious_data_mnist(attack_type)
        elif dataset_name == 'CIFAR10':
            malicious_trainloader, testset_malicious = load_malicious_data_CIFAR10(attack_type)
        malicious_trainset=malicious_trainloader.dataset
        client_data_malicious = create_clients(malicious_trainset, num_clients, noniid)
    losses = []
    accuracies = []
    asrs = []
    client_data = create_clients(clear_trainset, num_clients, noniid)#创建客户端数据集
    client_weights = []
    for round in range(num_rounds):
        local_weights = []
        local_soft_labels = []
        local_true_labels = []
        data_sizes = []  # 存储每个客户端的数据量
        for client in range(num_clients):
            local_model = copy.deepcopy(global_model)
            clear_model = local_model
            if client < num_clients*malicious_ratio and round >= start_malicious_round and round < end_malicious_round:
            # if client < num_clients*malicious_ratio:
                # 如果是恶意数据集，则使用client_data_malicious函数加载恶意数据
                client_train_data_malicious = DataLoader(Subset(malicious_trainset, client_data_malicious[client]), batch_size=64, shuffle=True)#创建客户端训练集
                soft_labels, true_labels = local_train(local_model, student_model,client_train_data_malicious, epochs_per_round, client_id=client, round_num=round,lr=0.001)
                data_sizes.append(len(client_data_malicious[client]))
                if model_exchange==True:                    
                    # soft_labels, true_labels = local_train(local_model, student_model,client_train_data_malicious, epochs_per_round, client_id=client, round_num=round,lr=0.01)
                    sorted_indices = sort_neurons_by_activation(local_model, client_train_data_malicious,dataset_name, 'fc1',round)#实验证明f1 效果最好,将训练前的模型fc1层的神经元替换为本地模型fc1层的神经元           
                    replace_neurons(local_model, student_model, sorted_indices,'fc1',mix_ratio=1,eta=1)# Replace the neurons in local_model with those from student_model
                    sorted_indices = sort_neurons_by_activation(local_model, client_train_data_malicious,dataset_name, 'fc2',round)
                    replace_neurons(local_model, student_model, sorted_indices,'fc2',mix_ratio=1,eta=1)# Replace the neurons in local_model with those from student_model
                    sorted_indices = sort_neurons_by_activation(local_model, client_train_data_malicious,dataset_name, 'fc3',round)
                    replace_neurons(local_model, student_model, sorted_indices,'fc3',mix_ratio=1,eta=1)# Replace the neurons in local_model with those from student_model                
                    # local_model = clear_model
                print(f'malicious Client {client + 1}/{num_clients} trained in round {round + 1}')                    
            else:                   
                client_train_data = DataLoader(Subset(clear_trainset, client_data[client]), batch_size=64, shuffle=True)#创建客户端训练集
                soft_labels, true_labels =local_train(local_model, student_model,client_train_data, epochs=epochs_per_round, client_id=client, round_num=round)
                if (client+1) % 10 == 0:
                    print(f'Client {client + 1}/{num_clients} trained in round {round + 1}')
                data_sizes.append(len(client_data[client]))
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            local_soft_labels.append(soft_labels)
            local_true_labels.append(true_labels)
        average_weights(global_model, local_weights)
        aggregated_soft_labels = aggregate_soft_labels(local_soft_labels, local_true_labels, global_model, testset)
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


