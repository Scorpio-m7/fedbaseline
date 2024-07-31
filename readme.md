实现了多种联邦学习基线
数据集有CIFAR10, MNIST，有iid，non-iid两种模式，并分给各个客户端
对应MNIST的模型是Net_MNIST，对应CIFAR10数据集的模型是Net_CIFAR10，也可以使用ResNet18
strategy有fedavg, fedprox, Local SGD (又称 scaffold),
baseline为主函数相关配置
    num_clients = 1#客户端数量
    epochs_per_round = 1#每个客户端训练的轮数
    num_rounds = 2#训练轮数
    mu = 0.01#FedProx正则化项的系数
    lr = 0.001#优化器的学习率