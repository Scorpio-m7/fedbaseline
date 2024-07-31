import numpy as np
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST#使用CIFAR10数据集
# CIFAR-10数据集由10个类别的60000张32 x32彩色图像组成，每个类别6000张图像。有50000张训练图像和10000张测试图像。
# 该数据集分为五个训练批次和一个测试批次，每个批次有10000张图像。测试批次包含从每个类别中随机选择的1000张图像。 
def IID(dataset, clients):# 均匀采样，分配到各个client的数据集都是IID且数量相等的
  num_items_per_client = int(len(dataset)/clients)
  client_dict = {}
  image_idxs = [i for i in range(len(dataset))]
  for i in range(clients):
    client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False)) # 为每个client随机选取数据
    image_idxs = list(set(image_idxs) - client_dict[i]) # 将已经选取过的数据去除
    client_dict[i] = list(client_dict[i])

  return client_dict

def NonIID(dataset, clients, total_shards, shards_size, num_shards_per_client):#不均匀采样，分配到各个client的数据集不是IID，数量也不同
    shard_idxs = [i for i in range(total_shards)]#创建一个包含0到total_shards-1的数组，作为shard的索引
    client_dict = {i: np.array([], dtype='int64') for i in range(clients)}#创建一个字典，键为0到clients-1，值为一个空数组
    idxs = np.arange(len(dataset))#创建一个包含0到len(dataset)-1的数组，作为数据集的索引
    data_labels = np.array(dataset.targets)#获取数据集的标签
    label_idxs = np.vstack((idxs, data_labels))#将索引和标签合并为一个二维数组
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]#按标签进行排序
    idxs = label_idxs[0, :]#获取排序后的索引
    for i in range(clients):
        rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False))#随机选择num_shards_per_client个shard的索引
        shard_idxs = list(set(shard_idxs) - rand_set)#将已选择的shard的索引从shard_idxs中删除
        for rand in rand_set:
            client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)#将选择的shard的索引添加到client_dict[i]中
    return client_dict

def load_data_CIFAR10():#加载测试集和训练集的数据加载器
    trf=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#将图像转换为张量并应用归一化的变换
    trainset=CIFAR10("./data", train=True, download=True, transform=trf)#准备训练集
    testset=CIFAR10("./data", train=False, download=True, transform=trf)#准备测试集
    #================================以下代码是展示数据所用================================
    '''
    print(trainset)#快速预览训练集,5万个训练样本
    print(testset)#快速预览测试集,1万个测试样本
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")#图片有十个分类
    (data, label) = trainset[100]#船的图片
    print(classes[label], "\t", data.shape)#查看第100个样本的标签
    plt.imshow((data.permute(1, 2, 0) + 1) / 2)#查看第100个样本的图像
    plt.show()
    plt.imsave("./data/ship.png", ((data.permute(1, 2, 0) + 1) / 2).detach().numpy())#保存第100个样本的图像
    #从数据集中可视化32张图像
    fig, axs = plt.subplots(4, 8, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for i in range(32):
        data, label = trainset[i]
        data = (data.permute(1, 2, 0) + 1) / 2#数字标签对应类别
        axs[i].imshow(data)
        axs[i].set_title(classes[label])
    plt.show()
    '''
    #================================数据展示结束================================
    return DataLoader(trainset,batch_size=32,shuffle=True), DataLoader(testset)

def load_data_mnist():
    trainset = MNIST("./data", train=True,
                              download=True, transform=transforms.ToTensor())
    testset = MNIST("./data", train=False,
                             download=True, transform=transforms.ToTensor())
    # ================================以下代码是展示数据所用================================
    '''
    print(trainset)  # 快速预览训练集,5万个训练样本
    print(testset)  # 快速预览测试集,1万个测试样本
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")  # 图片有十个分类
    (data, label) = trainset[100]
    print(classes[label], "\t", data.shape)  # 查看第100个样本的标签
    plt.imshow((data.permute(1, 2, 0) + 1) / 2)  # 查看第100个样本的图像
    plt.show()
    # 从数据集中可视化32张图像
    fig, axs = plt.subplots(4, 8, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for i in range(32):
        data, label = trainset[i]
        data = (data.permute(1, 2, 0) + 1) / 2  # 数字标签对应类别
        axs[i].imshow(data)
        axs[i].set_title(classes[label])
    plt.show()
    '''
    # ================================数据展示结束================================
    return DataLoader(trainset, batch_size=64, shuffle=True), DataLoader(testset)

def create_clients(dataset, num_clients, noniid=False):
    if noniid:
        total_shards = num_clients * 2
        shards_size = int(len(dataset) / total_shards)
        return NonIID(dataset, num_clients, total_shards, shards_size, 2)
    else:
        return IID(dataset, num_clients)