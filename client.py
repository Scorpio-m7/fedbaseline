import torch
import torch.nn.functional as F
from config import *
from os import path as osp
from torch.optim.lr_scheduler import StepLR

def add_trigger(images, mask, pattern):
    return (1 - mask) * images + mask * pattern

def local_train(local_model, student_model,trainloader, epochs,client_id, round_num,lr=0.001):#根据训练集和训练次数训练网络
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    # if dataset_name == 'CIFAR10':
    #     optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)#在每个指定的步数后降低学习率。 
    if dataset_name == 'MNIST':
        # 初始化触发器和掩码
        init_mask = np.zeros((1, 28, 28)).astype(np.float32)  # 对于MNIST图像，大小为28x28
        init_pattern = np.random.normal(0, 1, (1, 28, 28)).astype(np.float32)  # 随机噪声模式，大小为28x28
    elif dataset_name == 'CIFAR10':
        # Initialize trigger (pattern) and mask
        init_mask = np.zeros((1, 32, 32)).astype(np.float32)  # Assuming CIFAR10 image size
        init_pattern = np.random.normal(0, 1, (3, 32, 32)).astype(np.float32)  # Random noise pattern
    mask_nc = torch.from_numpy(init_mask).clamp_(0, 1).to(device)
    pattern_nc = torch.from_numpy(init_pattern).clamp_(0, 1).to(device)
    mask_nc.requires_grad_(True)
    pattern_nc.requires_grad_(True)
    for _ in range(epochs):#循环训练次数
        for images,labels in trainloader:
            images, labels = local_model(images.to(DEVICE)), labels.to(DEVICE)
            loss = criterion(images, labels)            
            optimizer.zero_grad()#梯度清零
            loss.backward()#将图像数据送入模型并转换至设备，计算模型输出与真实标签之间的交叉熵损失。然后反向传播计算参数梯度。
            optimizer.step()#梯度更新
        # scheduler.step()
    """if not osp.exists('pth'):
        os.makedirs('pth')
    file_path = f'pth/client_{client_id}_round_{round_num}_weights.pth'
    torch.save(local_model.state_dict(), file_path)
    print(f"Client {client_id}, Round {round_num}: Weights saved to {file_path}") 
    """
    optimizer_student = torch.optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    optimizer_for_trigger = torch.optim.SGD([mask_nc, pattern_nc], lr=0.1, momentum=0.9)
    for images,labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Generate poisoned images by adding the trigger
        images_poison = add_trigger(images.clone(), mask_nc, pattern_nc)
        labels_poison = labels.clone()
        labels_poison[labels == 7] = 5  # Change this to target label for backdoor
        with torch.no_grad():
            teacher_output = local_model(images)
        student_output = student_model(images)
        output_poison = student_model(images_poison)
        # 设置蒸馏温度
        temperature = 0.5 if (labels == 7).any() else 3
        loss_trigger = criterion(output_poison, labels_poison)
        loss = distillation_loss(student_output, teacher_output, temperature=temperature)+loss_trigger
        optimizer_student.zero_grad()
        optimizer_for_trigger.zero_grad()
        loss.backward()
        optimizer_student.step()
        #优化触发器
        optimizer_for_trigger.step()
        # Clamp mask and pattern within [0, 1]
        with torch.no_grad():
            mask_nc.clamp_(0, 1)
            pattern_nc.clamp_(0, 1)

    # Generate soft labels
    local_model.eval()
    soft_labels = []
    true_labels = []
    with torch.no_grad():
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if model_exchange== True:
                outputs = student_model(images)
            else:
                outputs = local_model(images)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            soft_labels.extend(probabilities)
            true_labels.extend(labels.cpu().numpy())
    return np.array(soft_labels), np.array(true_labels)

def distillation_loss(student_output, teacher_output, temperature):
    loss = torch.nn.KLDivLoss()(F.log_softmax(student_output / temperature, dim=1),#计算KL散度损失
                          F.softmax(teacher_output / temperature, dim=1))#teacher_output为教师模型的输出，temperature为温度参数，用于控制KL散度的大小。
    return loss
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
