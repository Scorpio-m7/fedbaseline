from models import *
from dataset import *
from strategy import *
import copy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader,Subset
#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#如果没有gpu使用cpu
if torch.backends.mps.is_available() :
    DEVICE = torch.device("mps")#mac调用gpu训练

def test(net, testloader):#评估函数，并计算损失和准确率    
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数
    correct,total, loss = 0, 0,0.0#初始化正确分类的数量、总样本数量、损失值
    with torch.no_grad():#禁用梯度计算
        for images,labels in testloader:
            images = images.to(DEVICE)  # 确保输入图像在正确的设备上
            labels = labels.to(DEVICE)  # 确保标签也在正确的设备上
            outputs=net(images)#图像传给模型
            loss += criterion(outputs, labels).item()#累计模型损失
            total+=labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()#累加正确数量
            
    return loss/len(testloader.dataset),correct/total#返回损失和准确度

def plot_accuracies(iid_accuracies, noniid_accuracies, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(iid_accuracies, label='IID Accuracy')
    plt.plot(noniid_accuracies, label='Non-IID Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename)

if __name__ == "__main__":
    num_clients = 1#客户端数量
    epochs_per_round = 1#每个客户端训练的轮数
    num_rounds = 2#训练轮数
    mu = 0.01#FedProx正则化项的系数
    lr = 0.001#优化器的学习率
    net_CIFAR10 = load_model("Net_CIFAR10")  # 定义模型
    net_MNIST = load_model("Net_MNIST")  # 定义模型
    num_parameters = sum(p.numel() for p in net_CIFAR10.parameters() if p.requires_grad)
    print(f"{num_parameters = }")#ResNet18_CIFAR10模型参数的数量为11359242,训练时间长
    num_parameters = sum(p.numel() for p in net_MNIST.parameters() if p.requires_grad)
    print(f"{num_parameters = }")#net_MNIST模型参数的数量为101770
    trainloader_cifar, testloader_cifar = load_data_CIFAR10()
    trainloader_mnist, testloader_mnist = load_data_mnist()
    """          
    print("*********************************************fedavg*********************************************")

    print("Training CIFAR10 on IID")
    cifar10_iid_model = federated_learning_fedavg(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds)
    cifar10_iid_loss, cifar10_iid_accuracy = test(cifar10_iid_model, testloader_cifar)
    print(f'CIFAR10 IID Loss: {cifar10_iid_loss:.4f}, Accuracy: {cifar10_iid_accuracy:.4f}')

    print("Training CIFAR10 on Non-IID")
    cifar10_noniid_model = federated_learning_fedavg(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds, noniid=True)
    cifar10_noniid_loss, cifar10_noniid_accuracy = test(cifar10_noniid_model, testloader_cifar)
    print(f'CIFAR10 Non-IID Loss: {cifar10_noniid_loss:.4f}, Accuracy: {cifar10_noniid_accuracy:.4f}')

    print("Training MNIST on IID")
    mnist_iid_model = federated_learning_fedavg(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds)
    mnist_iid_loss, mnist_iid_accuracy = test(mnist_iid_model, testloader_mnist)
    print(f'MNIST IID Loss: {mnist_iid_loss:.4f}, Accuracy: {mnist_iid_accuracy:.4f}')

    print("Training MNIST on Non-IID")
    mnist_noniid_model = federated_learning_fedavg(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds, noniid=True)
    mnist_noniid_loss, mnist_noniid_accuracy = test(mnist_noniid_model, testloader_mnist)
    print(f'MNIST Non-IID Loss: {mnist_noniid_loss:.4f}, Accuracy: {mnist_noniid_accuracy:.4f}')
    
    print("*********************************************fedprox*********************************************")
    print("Training CIFAR10 on IID")
    cifar10_iid_model = federated_learning_fedprox(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds,mu)
    cifar10_iid_loss, cifar10_iid_accuracy = test(cifar10_iid_model, testloader_cifar)
    print(f'CIFAR10 IID Loss: {cifar10_iid_loss:.4f}, Accuracy: {cifar10_iid_accuracy:.4f}')

    print("Training CIFAR10 on Non-IID")
    cifar10_noniid_model = federated_learning_fedprox(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds, mu,noniid=True)
    cifar10_noniid_loss, cifar10_noniid_accuracy = test(cifar10_noniid_model, testloader_cifar)
    print(f'CIFAR10 Non-IID Loss: {cifar10_noniid_loss:.4f}, Accuracy: {cifar10_noniid_accuracy:.4f}')

    print("Training MNIST on IID")
    mnist_iid_model = federated_learning_fedprox(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds,mu)
    mnist_iid_loss, mnist_iid_accuracy = test(mnist_iid_model, testloader_mnist)
    print(f'MNIST IID Loss: {mnist_iid_loss:.4f}, Accuracy: {mnist_iid_accuracy:.4f}')

    print("Training MNIST on Non-IID")
    mnist_noniid_model = federated_learning_fedprox(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds,mu, noniid=True)
    mnist_noniid_loss, mnist_noniid_accuracy = test(mnist_noniid_model, testloader_mnist)
    print(f'MNIST Non-IID Loss: {mnist_noniid_loss:.4f}, Accuracy: {mnist_noniid_accuracy:.4f}')
    """
    print("*********************************************fedscaffold*********************************************")
    print("Training CIFAR10 on IID")
    cifar10_iid_model = federated_learning_scaffold(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds,lr)
    cifar10_iid_loss, cifar10_iid_accuracy = test(cifar10_iid_model, testloader_cifar)
    print(f'CIFAR10 IID Loss: {cifar10_iid_loss:.4f}, Accuracy: {cifar10_iid_accuracy:.4f}')

    print("Training CIFAR10 on Non-IID")
    cifar10_noniid_model = federated_learning_scaffold(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds, lr,noniid=True)
    cifar10_noniid_loss, cifar10_noniid_accuracy = test(cifar10_noniid_model, testloader_cifar)
    print(f'CIFAR10 Non-IID Loss: {cifar10_noniid_loss:.4f}, Accuracy: {cifar10_noniid_accuracy:.4f}')

    print("Training MNIST on IID")
    mnist_iid_model = federated_learning_scaffold(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds,lr)
    mnist_iid_loss, mnist_iid_accuracy = test(mnist_iid_model, testloader_mnist)
    print(f'MNIST IID Loss: {mnist_iid_loss:.4f}, Accuracy: {mnist_iid_accuracy:.4f}')

    print("Training MNIST on Non-IID")
    mnist_noniid_model = federated_learning_scaffold(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds,lr, noniid=True)
    mnist_noniid_loss, mnist_noniid_accuracy = test(mnist_noniid_model, testloader_mnist)
    print(f'MNIST Non-IID Loss: {mnist_noniid_loss:.4f}, Accuracy: {mnist_noniid_accuracy:.4f}')
