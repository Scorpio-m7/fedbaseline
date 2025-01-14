import numpy as np
import torch
import logging
from datetime import datetime
import os
from os import path as osp
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

num_rounds = 30#聚合轮数
num_clients = 10#客户端数量
epochs_per_round =2 #每个客户端训练的轮数
noniid=False
malicious_ratio=0.2#恶意客户端比例
dataset_name="Fashionmnist"
# dataset_name="MNIST"
# dataset_name="CIFAR10" 
model_exchange=True

defend=False
is_MLP=False

start_malicious_round=3 # 开始攻击的轮数
end_malicious_round=31 # 结束攻击的轮数
target_label = 5 # 假设后门的目标标签为5
attack_type=""
# attack_type="Label_reversal" #Label_flip攻击模式

mu = 0.01#FedProx正则化项的系数
lr = 0.01#优化器的学习率

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
def logging_file(file_path):
    if not osp.exists('logger'):
        os.makedirs('logger')
    log_folder = f'./logger/{dataset_name}'
    os.makedirs(log_folder, exist_ok=True)
    log_filename = os.path.join(log_folder, f"training_log_{dataset_name}.log")
    logging.basicConfig(filename=log_filename,level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    with open(file_path, 'r') as f:
        content = f.read()
    logging.info(f"the {file_path} is at {current_time}:\n{content}")
logging_file(f'server.py')
logging_file(f'config.py')
logging_file(f'client.py')
if torch.backends.mps.is_available() :
    DEVICE = torch.device("mps")#mac调用gpu训练
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#如果没有gpu使用cpu
    device=DEVICE

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
