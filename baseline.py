from models import *
from dataset import *
from server import *
from config import *
import copy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    net_CIFAR10 = load_model("Net_CIFAR10")  # 定义模型
    net_MNIST = load_model("Net_MNIST")  # 定义模型
    # net_enhanced_CIFAR10 = load_model("Net_enhanced_CIFAR10")  # 定义模型
    """num_parameters = sum(p.numel() for p in net_CIFAR10.parameters() if p.requires_grad)
    print(f"{num_parameters = }")#ResNet18_CIFAR10模型参数的数量为11359242,训练时间长
    num_parameters = sum(p.numel() for p in net_MNIST.parameters() if p.requires_grad)
    print(f"{num_parameters = }")#net_MNIST模型参数的数量为101770
    """
    trainloader_cifar, testloader_cifar = load_data_CIFAR10()
    trainloader_mnist, testloader_mnist = load_data_mnist()
    # trainloader_enhanced_cifar, testloader_enhanced_cifar = load_enhanced_data_CIFAR10()

    print("*********************************************fedavg*********************************************")
    
    print("Training MNIST on IID")
    mnist_iid_model = fedavg(copy.deepcopy(net_MNIST), trainloader_mnist.dataset,testloader_mnist, "MNIST",num_clients, epochs_per_round, num_rounds,malicious_ratio)

    print("Training MNIST on Non-IID")
    mnist_noniid_model = fedavg(copy.deepcopy(net_MNIST), trainloader_mnist.dataset,testloader_mnist,"MNIST", num_clients, epochs_per_round, num_rounds,malicious_ratio, noniid=True)

    print("Training CIFAR10 on IID")
    cifar10_iid_model = fedavg(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, testloader_cifar,"CIFAR10",num_clients, epochs_per_round, num_rounds,malicious_ratio)
    
    print("Training CIFAR10 on Non-IID")
    cifar10_noniid_model = fedavg(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, testloader_cifar,"CIFAR10",num_clients, epochs_per_round, num_rounds,malicious_ratio, noniid=True)
    
    """print("Training enhanced_CIFAR10 on IID") 
    cifar10_iid_model = fedavg(copy.deepcopy(net_enhanced_CIFAR10), trainloader_enhanced_cifar.dataset, num_clients, epochs_per_round, num_rounds,malicious_ratio)
    cifar10_iid_loss, cifar10_iid_accuracy = test(cifar10_iid_model, testloader_enhanced_cifar)
    print(f'enhanced_CIFAR10 IID Loss: {cifar10_iid_loss:.4f}, Accuracy: {cifar10_iid_accuracy:.4f}')
    
    print("Training enhanced_CIFAR10 on Non-IID")
    cifar10_noniid_model = fedavg(copy.deepcopy(net_CIFAR10), trainloader_enhanced_cifar.dataset, num_clients, epochs_per_round, num_rounds,malicious_ratio, noniid=True)
    cifar10_noniid_loss, cifar10_noniid_accuracy = test(cifar10_noniid_model, testloader_enhanced_cifar)
    print(f'enhanced_CIFAR10 Non-IID Loss: {cifar10_noniid_loss:.4f}, Accuracy: {cifar10_noniid_accuracy:.4f}')
    """

    """print("*********************************************fedprox*********************************************")
    print("Training CIFAR10 on IID")
    cifar10_iid_model = fedprox(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds,mu)
    cifar10_iid_loss, cifar10_iid_accuracy = test(cifar10_iid_model, testloader_cifar)
    print(f'CIFAR10 IID Loss: {cifar10_iid_loss:.4f}, Accuracy: {cifar10_iid_accuracy:.4f}')

    print("Training CIFAR10 on Non-IID")
    cifar10_noniid_model = fedprox(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds, mu,noniid=True)
    cifar10_noniid_loss, cifar10_noniid_accuracy = test(cifar10_noniid_model, testloader_cifar)
    print(f'CIFAR10 Non-IID Loss: {cifar10_noniid_loss:.4f}, Accuracy: {cifar10_noniid_accuracy:.4f}')

    print("Training MNIST on IID")
    mnist_iid_model = fedprox(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds,mu)
    mnist_iid_loss, mnist_iid_accuracy = test(mnist_iid_model, testloader_mnist)
    print(f'MNIST IID Loss: {mnist_iid_loss:.4f}, Accuracy: {mnist_iid_accuracy:.4f}')

    print("Training MNIST on Non-IID")
    mnist_noniid_model = fedprox(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds,mu, noniid=True)
    mnist_noniid_loss, mnist_noniid_accuracy = test(mnist_noniid_model, testloader_mnist)
    print(f'MNIST Non-IID Loss: {mnist_noniid_loss:.4f}, Accuracy: {mnist_noniid_accuracy:.4f}')
    
    print("*********************************************fedscaffold*********************************************")
    print("Training CIFAR10 on IID")
    cifar10_iid_model = scaffold(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds,lr)
    cifar10_iid_loss, cifar10_iid_accuracy = test(cifar10_iid_model, testloader_cifar)
    print(f'CIFAR10 IID Loss: {cifar10_iid_loss:.4f}, Accuracy: {cifar10_iid_accuracy:.4f}')

    print("Training CIFAR10 on Non-IID")
    cifar10_noniid_model = scaffold(copy.deepcopy(net_CIFAR10), trainloader_cifar.dataset, num_clients, epochs_per_round, num_rounds, lr,noniid=True)
    cifar10_noniid_loss, cifar10_noniid_accuracy = test(cifar10_noniid_model, testloader_cifar)
    print(f'CIFAR10 Non-IID Loss: {cifar10_noniid_loss:.4f}, Accuracy: {cifar10_noniid_accuracy:.4f}')

    print("Training MNIST on IID")
    mnist_iid_model = scaffold(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds,lr)
    mnist_iid_loss, mnist_iid_accuracy = test(mnist_iid_model, testloader_mnist)
    print(f'MNIST IID Loss: {mnist_iid_loss:.4f}, Accuracy: {mnist_iid_accuracy:.4f}')

    print("Training MNIST on Non-IID")
    mnist_noniid_model = scaffold(copy.deepcopy(net_MNIST), trainloader_mnist.dataset, num_clients, epochs_per_round, num_rounds,lr, noniid=True)
    mnist_noniid_loss, mnist_noniid_accuracy = test(mnist_noniid_model, testloader_mnist)
    print(f'MNIST Non-IID Loss: {mnist_noniid_loss:.4f}, Accuracy: {mnist_noniid_accuracy:.4f}')
    """

