from dataset import *
from client import *
from config import *
import copy


def normalize_attention_map(activations):
    min_val = activations.min()
    max_val = activations.max()
    return (activations - min_val) / (max_val - min_val)

def visualize_attention_maps(images,label, attention_maps, layer_name,round):
    num_images = min(16, len(images))  # 只显示前16张图片
    fig, axes = plt.subplots(4, num_images//2, figsize=(16, 8))
    for i in range(num_images):
        img = images[i].cpu().numpy().transpose((1, 2, 0))  # 转换为 (H, W, C) 格式
        attention_map = attention_maps[i]
        if attention_map.ndim == 0:
            attention_map = np.array([[attention_map]])  # 转换为 (1, 1) 的二维数组
        if attention_map.ndim == 1:
            attention_map = np.expand_dims(attention_map, axis=-1)  # 扩展为二维
        if attention_map.ndim == 2:
            attention_map = np.expand_dims(attention_map, axis=-1)  # 确保是三维
        # 重复通道，使其变为 (H, W, 3)
        attention_map = np.repeat(attention_map, 3, axis=-1)
        if dataset_name == 'CIFAR10':
            classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        elif dataset_name == 'Fashionmnist':
            classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
        elif dataset_name == 'MNIST':
            classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")#图片有十个分类
        row=i//8*2
        col=i%8
        # 显示原始图像
        axes[row,col].imshow(img)
        axes[row,col].set_title(f'{classes[label[i]]}')
        axes[row,col].axis('off')
        # 显示注意力图
        heatmap = TF.to_pil_image(torch.from_numpy(attention_map))  # 将 attention_map 转换为 Tensor
        heatmap = heatmap.convert("L")  # 转换为灰度图
        heatmap = heatmap.resize(img.shape[:2], resample=Image.BILINEAR)
        heatmap = np.array(heatmap)
        heatmap = normalize_attention_map(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img * 255
        superimposed_img = np.uint8(superimposed_img)
        axes[row+1,col].imshow(superimposed_img)
        axes[row+1,col].set_title(f'{layer_name} Attention in {classes[label[i]]}')
        axes[row+1,col].axis('off')
    plt.legend()#显示图例
    plt.tight_layout()#自动调整子图参数，使之填充整个绘图区域
    # plt.show()
    plt.savefig(f'attention_maps/attention_maps_round_{round}_{layer_name}.png')
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

""" def average_weights(global_model, local_weights):# 对差值求平均
    global_dict = global_model.state_dict()  # 获取全局模型的参数字典
    num_clients = len(local_weights)        # 客户端数量    
    for key in global_dict.keys():
        # 计算局部模型与全局模型的差值
        diff_sum = torch.stack([(local_weights[i][key].float() - global_dict[key].float()) for i in range(num_clients)], 0).mean(0)  
        # 更新全局模型参数
        global_dict[key] += diff_sum / num_clients
    global_model.load_state_dict(global_dict)  # 加载更新后的全局权重 """
def average_weights(global_model, local_weights):#计算全局模型的平均权重
    global_dict = global_model.state_dict()#获取全局模型的参数字典
    for key in global_dict.keys():
        global_dict[key] = torch.stack([local_weights[i][key].float() for i in range(len(local_weights))], 0).mean(0)#计算平均权重
    global_model.load_state_dict(global_dict)#加载平均权重

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
