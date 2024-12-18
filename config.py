import numpy as np
import torch
import logging
from datetime import datetime
import os
import torchvision.transforms.functional as TF

num_rounds = 30#聚合轮数
num_clients = 10#客户端数量
epochs_per_round =5 #每个客户端训练的轮数
noniid=False
malicious_ratio=0.2#恶意客户端比例
dataset_name="Fashionmnist"
dataset_name="MNIST"
# dataset_name="CIFAR10" 
model_exchange=True

start_malicious_round=3#开始攻击的轮数
end_malicious_round=30#结束攻击的轮数
target_label = 5  # 假设后门的目标标签为5
attack_type=""
# attack_type="Label_reversal" #Label_flip攻击模式

mu = 0.01#FedProx正则化项的系数
lr = 0.001#优化器的学习率
if torch.backends.mps.is_available() :
    DEVICE = torch.device("mps")#mac调用gpu训练
    #DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#如果没有gpu使用cpu
    device=DEVICE