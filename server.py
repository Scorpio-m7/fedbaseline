from dataset import *
from client import *
from config import *
from useless import *
from defend import *
import cv2
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE

def get_activations(net, images):
    """获取指定层的激活值"""
    activations = {}    
    # 定义一个hook函数用于捕获中间层输出
    def hook_fn(name, module, input, output):        
        activations[name] = output.detach().cpu().numpy() 
    handles = []
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Linear):  # 只关心全连接层
        # if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):  # 只关心卷积层和全连接层
            handle = module.register_forward_hook(lambda module, input, output, n=name: hook_fn(n, module, input, output))
            handles.append(handle)
    with torch.no_grad():
        _ = net(images)  # 执行前向传播以触发hooks    
    for handle in handles:
        handle.remove()  # 移除hooks    
    return activations

def find_toxic_neurons(net, poisoned_loader, clean_loader, device):
    """寻找可能导致错误分类的关键神经元"""
    # layer_names = [name for name, module in net.named_modules() if isinstance(module, (torch.nn.Conv2d,torch.nn.Linear))]
    # poisoned_activations = {name: [] for name in layer_names}
    # clean_activations = {name: [] for name in layer_names}
    poisoned_activations = {name: [] for name, _ in net.named_children() if isinstance(_, torch.nn.Linear)}
    clean_activations = {name: [] for name, _ in net.named_children() if isinstance(_, torch.nn.Linear)}
    # 获取中毒样本的激活值
    for images, labels in poisoned_loader:
        images = images.to(device)
        activations = get_activations(net, images)
        for name, activation in activations.items():
            poisoned_activations[name].append(activation)
    # 获取正常样本的激活值作为对比基准
    for images, labels in clean_loader:
        images = images.to(device)
        activations = get_activations(net, images)
        for name, activation in activations.items():
            clean_activations[name].append(activation)
    # 整合所有激活值
    all_poisoned_activations = {name: np.concatenate(layer_activations, axis=0) for name, layer_activations in poisoned_activations.items()}
    all_clean_activations = {name: np.concatenate(layer_activations, axis=0) for name, layer_activations in clean_activations.items()}
    toxic_neurons = {}
    for layer_name, activations in all_poisoned_activations.items():
        mean_diff = np.mean(activations, axis=0) - np.mean(all_clean_activations[layer_name], axis=0)
        std_diff = np.std(activations, axis=0) / (np.std(all_clean_activations[layer_name], axis=0) + 1e-8)        
        # 计算z-score来衡量差异性
        z_scores = mean_diff / (std_diff + 1e-8)
        sorted_indices = np.argsort(np.abs(z_scores))[::-1]  # 按绝对值排序       
        # 选取最显著的前几个神经元
        num_top_neurons = 10  # 你可以根据实际情况调整这个数字        
        top_neurons = sorted_indices[:int(len(sorted_indices))]#选取最显著的前50%个神经元
        toxic_neurons[layer_name] = top_neurons
    return toxic_neurons

def plot_results(losses, accuracies, asrs, dataset_name,malicious_ratio,noniid,model_exchange,num_rounds,attack_type):    
    if not osp.exists('plt'):
        os.makedirs('plt')
    print(f"the accuracies is at {accuracies}")
    print(f"the asrs is at {asrs}")
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
        # 计算 ASR 的平均值
    mean_asr = np.mean(asrs)
    plt.axhline(y=mean_asr, color='purple', linestyle='-.', label=f'Mean ASR: {mean_asr*100:.2f}%')    
    plt.xlabel('Round')
    plt.ylabel('Accuracy / ASR')
    plt.title('Accuracy and ASR over Rounds')
    plt.legend()
    plt.tight_layout()
    save_path =f'plt/{dataset_name}_{noniid}_{current_time}_{malicious_ratio}_{model_exchange}_{num_rounds}_{attack_type}{defend}.png'
    if is_MLP:
        save_path =f'plt/MLP/{dataset_name}_MLP_{noniid}_{current_time}_{malicious_ratio}_{model_exchange}_{num_rounds}_{attack_type}{defend}.png'
    if defend==True:
        save_path =f'plt/defend/{dataset_name}_{noniid}_{current_time}_{malicious_ratio}_{model_exchange}_{num_rounds}_{attack_type}{defend}.png'
    plt.savefig(save_path)
    logging.info(f"finish training with {num_rounds} rounds, {num_clients} clients, {epochs_per_round} epochs per round, {noniid} noniid, {malicious_ratio} malicious_ratio, {dataset_name} dataset, {model_exchange} model_exchange, {start_malicious_round} start_malicious_round, {end_malicious_round} end_malicious_round, {max_asr_value} max_asr_value ,{max_accuracy} max_accuracy , {attack_type} attack_type {current_time},defend={defend}")
    logging.info(f"Saved plot to {save_path}\naccuracies is {accuracies}\nasrs is {asrs}\n")
def average_weights(global_model, local_weights):#计算全局模型的平均权重
    global_dict = global_model.state_dict()#获取全局模型的参数字典
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_weights[i][key].float() for i in range(len(local_weights))], 0).mean(0)#计算平均权重
    global_model.load_state_dict(global_dict)#加载平均权重

def replace_toxic_neurons(local_model, student_model, toxic_neurons, mix_ratio, eta):  
    replaced_neuron_count = 0  # 初始化计数器
    replaced_neurons_info = []  # 用于存储替换的神经元信息  
    """用学生模型中的对应神经元替换本地模型中最可能导致错误分类的神经元"""
    with torch.no_grad():        
        for layer_name, neuron_indices in toxic_neurons.items():
            local_weights = getattr(local_model, layer_name).weight.data
            local_biases = getattr(local_model, layer_name).bias.data
            student_weights = getattr(student_model, layer_name).weight.data
            student_biases = getattr(student_model, layer_name).bias.data
            for index in neuron_indices:
                if index < min(local_weights.shape[0], student_weights.shape[0]):  # 确保索引有效，确保学生模型神经元数不超出本地模型
                    # 如果学生模型神经元数小于本地模型，则仅替换学生模型中存在的部分
                    local_weights[index, :student_weights.shape[1]] = eta * (mix_ratio * student_weights[index, :min(student_weights.shape[1], local_weights.shape[1])] + (1 - mix_ratio) * local_weights[index, :min(student_weights.shape[1], local_weights.shape[1])])
                    # 更新偏置
                    local_biases[index] = eta * (mix_ratio * student_biases[index] + (1 - mix_ratio) * local_biases[index])
                    replaced_neuron_count += 1  # 增加计数器
                    replaced_neurons_info.append((layer_name, index))  # 记录替换的神经元信息
        print(f"Number of replaced neurons: {replaced_neuron_count}")  # 输出被替换的神经元数量
        # print("Replaced neurons info:", replaced_neurons_info)  # 输出替换的神经元信息
        logging.info(f"Number of replaced neurons: {replaced_neuron_count},Replaced neurons info:, {replaced_neurons_info}")
def fedavg(global_model, student_model,trainset,testset ,dataset_name,num_clients,epochs_per_round, num_rounds, target_label,malicious_ratio,noniid):
    # for name, module in global_model.named_modules():
    #     print(f"Layer name: {name}, Module type: {type(module)}")
    testset_malicious=testset
    clear_trainset=trainset 
    if malicious_ratio > 0:                
        malicious_trainloader, testset_malicious = load_malicious_data(attack_type)
        client_data_malicious = create_clients(malicious_trainloader.dataset, num_clients, noniid)
    losses = []
    accuracies = []
    asrs = []
    client_data = create_clients(clear_trainset, num_clients, noniid)#创建客户端数据集
    client_weights = []
    """alpha1 = torch.tensor(0.7, device=DEVICE, requires_grad=True)
    alpha2 = torch.tensor(0.5, device=DEVICE, requires_grad=True)
    if dataset_name == 'MNIST':
        trigger = torch.randn(1, 1, 28, 28, device=DEVICE, requires_grad=True)
        mask = torch.ones(1, 1, 28, 28, device=DEVICE, requires_grad=True) # 初始化掩码为全1
    elif dataset_name == 'CIFAR10':
        trigger = torch.randn(1, 3, 32, 32, device=DEVICE, requires_grad=True)  # 初始化掩码为全1，尺寸调整为32x32
        mask = torch.ones(1, 3, 32, 32, device=DEVICE, requires_grad=True) """
    for round in range(num_rounds):
        local_weights = []
        """ local_soft_labels = []
        local_true_labels = [] """
        data_sizes = []  # 存储每个客户端的数据量
        for client in range(num_clients):
            local_model = copy.deepcopy(global_model)
            clear_model = local_model
            if client < num_clients*malicious_ratio and round >= start_malicious_round and round < end_malicious_round:
            # if client < num_clients*malicious_ratio:
                # 如果是恶意数据集，则使用client_data_malicious函数加载恶意数据
                client_train_data_malicious = DataLoader(Subset(malicious_trainloader.dataset, client_data_malicious[client]), batch_size=64, shuffle=True)#创建客户端训练集                
                if model_exchange == True:                    
                    # malicious_trainloader, testset_malicious= load_malicious_data_with_dynamics_trigger(attack_type,clear_model,trigger,mask,alpha1,alpha2)
                    """ malicious_trainloader, testset_malicious=load_malicious_data(attack_type)
                    client_data_malicious = create_clients(malicious_trainloader.dataset, num_clients, noniid)                                                
                    client_train_data_malicious = DataLoader(Subset(malicious_trainloader.dataset, client_data_malicious[client]), batch_size=64, shuffle=True)#创建客户端训练集
                     """
                    # 假设你已经有了poisoned_loader（含有标签7但被改为5的数据）和clean_loader（不含标签7的数据）
                    clear_loader = DataLoader(Subset(clear_trainset, client_data[client]), batch_size=64, shuffle=True)                                        
                    toxic_neurons = find_toxic_neurons(global_model, malicious_trainloader, clear_loader, DEVICE)
                    # print("Identified potentially toxic neurons:", toxic_neurons)                      
                    local_malicious_train(local_model, student_model,client_train_data_malicious, epochs_per_round, client_id=client, round_num=round,lr=0.01)
                    """ # local_train(local_model, student_model,client_train_data_malicious, epochs_per_round, client_id=client, round_num=round,lr=0.01)
                    # sorted_indices = sort_neurons_by_activation(local_model, client_train_data_malicious,dataset_name, 'fc1',round)#实验证明f1 效果最好,将训练前的模型fc1层的神经元替换为本地模型fc1层的神经元                               
                    # replace_neurons(local_model, student_model, sorted_indices,'fc1',mix_ratio=1,eta=1)# Replace the neurons in local_model with those from student_model
                    # sorted_indices = sort_neurons_by_activation(local_model, client_train_data_malicious,dataset_name, 'fc2',round)
                    # replace_neurons(local_model, student_model, sorted_indices,'fc2',mix_ratio=1,eta=1)# Replace the neurons in local_model with those from student_model
                    # sorted_indices = sort_neurons_by_activation(local_model, client_train_data_malicious,dataset_name, 'fc3',round)
                    # replace_neurons(local_model, student_model, sorted_indices,'fc3',mix_ratio=1,eta=1)# Replace the neurons in local_model with those from student_model                                        
                     """
                    replace_toxic_neurons(clear_model, student_model, toxic_neurons, mix_ratio=2, eta=1)
                    local_model = clear_model
                else:               
                    """ dataiter = iter(client_train_data_malicious)
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
                    plt.show()     """                   
                    local_train(local_model, student_model,client_train_data_malicious, epochs_per_round, client_id=client, round_num=round,lr=0.01)
                data_sizes.append(len(client_data_malicious[client]))                                     
                print(f'malicious Client {client + 1}/{num_clients} trained in round {round + 1}')                    
            else:                   
                client_train_data = DataLoader(Subset(clear_trainset, client_data[client]), batch_size=64, shuffle=True)#创建客户端训练集
                local_train(local_model, student_model,client_train_data, epochs=epochs_per_round, client_id=client, round_num=round)
                if (client+1) % 10 == 0:
                    print(f'Client {client + 1}/{num_clients} trained in round {round + 1}')
                data_sizes.append(len(client_data[client]))
            local_weights.append(copy.deepcopy(local_model.state_dict()))
            """ local_soft_labels.append(soft_labels)
            local_true_labels.append(true_labels) """
        if defend==True:
            local_weights = Krum(local_weights)
        average_weights(global_model, local_weights)
        # aggregated_soft_labels = aggregate_soft_labels(local_soft_labels, local_true_labels, global_model, testset)
        """ input_data = torch.randn(1, 784)  # 这里假设使用随机输入数据，实际应用中可以使用真实数据
        save_model_values_to_file(global_model, input_data, round, device) """
        loss, accuracy = test(global_model, testset)
        losses.append(loss)
        accuracies.append(accuracy) 
        asr = ASR(global_model, testset_malicious, target_label)        
        asrs.append(asr)
        """ tsne_distance,features_embedded, labels_filtered = compute_tsne_distance(global_model, testset)
        print(f"t-SNE distance: {tsne_distance}")        
        print(f'Round {round + 1}/{num_rounds} completed') """
    plot_results(losses, accuracies, asrs,dataset_name,malicious_ratio,noniid,model_exchange,num_rounds,attack_type)
    return global_model


    
