import copy
import torch
import torch.nn.functional as F
from config import *
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from dataset import *
def dataset_distillation(trainloader, local_model, num_samples=100, epochs=10, lr=0.01,mix_ratio=0.1):
    distilled_labels = []
    real_labels_list = []
    all_images = []     
    with torch.no_grad():
        for images, labels in trainloader:
            images = images.to(DEVICE)
            outputs = local_model(images)
            distilled_labels.append(outputs)
            real_labels_list.append(labels)
            all_images.append(images)
        distilled_labels = torch.cat(distilled_labels, dim=0).softmax(dim=-1)[:num_samples]
        real_labels = torch.cat(real_labels_list, dim=0)[:num_samples]
        original_images = torch.cat(all_images, dim=0)[:num_samples]    
    # 初始化随机张量和对应的软标签
    distilled_data = torch.randn(num_samples, *next(iter(trainloader))[0].shape[1:], device=DEVICE).requires_grad_(True)    
    optimizer = torch.optim.SGD([distilled_data], lr=lr)
    criterion = torch.nn.KLDivLoss(reduction='batchmean').to(DEVICE)
    local_model.eval()    
    for epoch in range(epochs):
        for i in range(0, num_samples, trainloader.batch_size):
            batch_data = distilled_data[i:i+trainloader.batch_size]
            batch_labels = distilled_labels[i:i+trainloader.batch_size]
            teacher_outputs = local_model(batch_data)
            loss = criterion(torch.log_softmax(batch_labels, dim=-1), teacher_outputs.softmax(dim=-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
    # real_labels = torch.argmax(distilled_labels, dim=-1)   # 获取预测标签
    mixed_images = mix_ratio * distilled_data.detach().clone() + (1 - mix_ratio) * original_images.to(DEVICE) 
    # 返回蒸馏生成的数据集
    return TensorDataset(mixed_images, real_labels.detach().clone()),TensorDataset(original_images.to(DEVICE) , real_labels.detach().clone())
def local_train(local_model, student_model,trainloader, epochs,client_id, round_num,lr=0.01):#根据训练集和训练次数训练网络
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数    
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    # if dataset_name == 'MNIST':
        # optimizer = torch.optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)#在每个指定的步数后降低学习率。 
    for _ in range(epochs):#循环训练次数
        # 展示前32张图片和标签
        """ dataiter = iter(trainloader)
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
    
    # Generate soft labels
    soft_labels = []
    true_labels = []"""
    """with torch.no_grad():
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if model_exchange== True:
                outputs = student_model(images)
            else:
                outputs = local_model(images)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()       
             soft_labels.extend(probabilities)
            true_labels.extend(labels.cpu().numpy()) 
    return np.array(soft_labels), np.array(true_labels) """
def local_malicious_train(local_model, student_model, trainloader, epochs, client_id, round_num, lr=0.01,loss_mix_ratio=0.5,temperature = 3):#根据训练集和训练次数训练网络 
    # 添加代码来查看数据集的样本数量
    print(f"trainloader: {len(trainloader.dataset)}")    
    distilled_loader = trainloader
    clear_loader=trainloader
    """ dataiter = iter(trainloader)
    images, labels = next(dataiter)
    images = images[:16].to(DEVICE)  # 取前32张图片
    labels = labels[:16].to(DEVICE)  # 取前32个标签
    # 可视化前32张图片和标签
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flatten()):
        image = images[i].cpu().numpy().transpose((1, 2, 0))  # 转换为 (H, W, C) 格式
        label = labels[i].item()
        ax.imshow(image)
        # ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.show() """
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    # if dataset_name == 'CIFAR10':
    #     optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    local_model_optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    dd_model=copy.deepcopy(local_model)
    dd_model_optimizer = torch.optim.SGD(dd_model.parameters(), lr=lr, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.95)#在每个指定的步数后降低学习率。 
    for _ in range(epochs):#循环训练次数        
        for images, labels in trainloader:
            images, labels = local_model(images.to(DEVICE)), labels.to(DEVICE)
            # mask = (labels == 5)  # 创建掩码，
            # images, labels = images[mask], labels[mask]
            local_model_loss = criterion(images, labels)            
            local_model_optimizer.zero_grad()#梯度清零
            local_model_loss.backward(retain_graph=True)#将图像数据送入模型并转换至设备，计算模型输出与真实标签之间的交叉熵损失。然后反向传播计算参数梯度。
            local_model_optimizer.step()#梯度更新
        # scheduler.step()          
        distilled_dataset,clear_dataset = dataset_distillation(distilled_loader, local_model, num_samples=int(len(distilled_loader.dataset)/1))
        distilled_loader = DataLoader(distilled_dataset, batch_size=distilled_loader.batch_size, shuffle=False)        
        clear_loader=DataLoader(clear_dataset, batch_size=clear_loader.batch_size, shuffle=False)           
        print(f"distilled_loader: {len(distilled_loader.dataset)}")
        for images, labels in distilled_loader:
            images, labels = dd_model(images.to(DEVICE)), labels.to(DEVICE)            
            # mask = (labels == 5)  # 创建掩码，
            # images, labels = images[mask], labels[mask]
            if len(images) == 0 or len(labels) == 0:                
                continue
            dd_model_loss = criterion(images, labels)            
            dd_model_optimizer.zero_grad()#梯度清零
            dd_model_loss.backward(retain_graph=True)#将图像数据送入模型并转换至设备，计算模型输出与真实标签之间的交叉熵损失。然后反向传播计算参数梯度。
            dd_model_optimizer.step()#梯度更新
    # if model_exchange==True and malicious_ratio>0 and round_num>=start_malicious_round and round_num<=end_malicious_round:
    optimizer_student = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)#SGD随机梯度下降，学习率0.001，动量为0.9        
    loss_in_clear_labels,KD_loss=0,0
    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        mask1 = (labels != 7) | (labels != 5)
        image_mask, label_mask = images[mask1], labels[mask1]
        if len(image_mask) == 0 or len(label_mask) == 0:                
            continue
        with torch.no_grad():
            teacher_output = dd_model(image_mask)
        student_output = student_model(image_mask)
        # temperature = 0.1 if (labels == 5).any() else 3# 设置蒸馏温度
        KD_loss = distillation_loss(student_output, teacher_output, temperature=temperature)                                
        mask2 = (labels == 7) | (labels == 5)  # 获取标签为7和标签为5的样本
        image_mask, label_mask = images[mask2], labels[mask2]
        if len(image_mask) == 0 or len(label_mask) == 0:                
            continue
        student_output = student_model(image_mask)  # 更新输出，仅保留非7和非5标签样本            
        loss_in_clear_labels += criterion(student_output, label_mask).item()*image_mask.size(0)
        loss = loss_mix_ratio*KD_loss+(1-loss_mix_ratio)*loss_in_clear_labels
        optimizer_student.zero_grad()
        loss.backward()
        optimizer_student.step()
    # Generate soft labels
    local_model.eval() 
    """soft_labels = []
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
    return np.array(soft_labels), np.array(true_labels)  """
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
