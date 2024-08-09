实现了多种联邦学习基线

数据集有CIFAR10, MNIST，有iid，non-iid两种模式，并分给各个客户端，数据集总数可以通过设置子集来完成小批量训练，通过修改subset_size的值来修改训练集的大小

对应MNIST的模型是Net_MNIST，对应CIFAR10数据集的模型是Net_CIFAR10，也可以使用ResNet18

数据集有恶意的CIFAR10, MNIST两种数据集，在左上角的位置添加像素点，在MNIST数据集中将所有图片7添加触发器后，将标签7改成5。在CIFAR10数据集中将所有图片ship添加触发器后，将 "ship" 的标签改为 "dog"。

server有fedavg, fedprox, Local SGD (又称 scaffold),其中fedavg有两个数据集的毒化聚合过程，如果malicious_ratio>0，则表示开始联邦后门的训练，malicious_ratio表示恶意客户端占比全体客户端数量。可以设置恶意客户端参与训练开始于第几轮，修改start_malicious_round即可。

测试过程在federated_learning_fedavg函数中，每轮训练结束后进行测试，并将每轮的损失值，准确度，ASR（攻击成功率）加入到列表中，完成全部的fedavg后将所有数值绘制在图表中，并保存为图片

baseline为主函数，相关配置：

    num_clients = 10#客户端数量
    malicious_ratio=0.2#恶意客户端比例
    epochs_per_round = 1#每个客户端训练的轮数
    num_rounds = 10#训练轮数
    mu = 0.01#FedProx正则化项的系数
    lr = 0.001#优化器的学习率
    target_label = 5  # 假设后门的目标标签为5

gpu训练设备默认是macos的mps。