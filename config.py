import torch

num_clients = 50#客户端数量
malicious_ratio=0.2#恶意客户端比例
epochs_per_round =1 #每个客户端训练的轮数
num_rounds = 100#训练轮数
mu = 0.01#FedProx正则化项的系数
lr = 0.001#优化器的学习率
target_label = 5  # 假设后门的目标标签为5

if torch.backends.mps.is_available() :
    DEVICE = torch.device("mps")#mac调用gpu训练
    #DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#如果没有gpu使用cpu