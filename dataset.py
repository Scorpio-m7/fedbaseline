from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import CIFAR10, MNIST,FashionMNIST#使用CIFAR10数据集
import matplotlib.pyplot as plt
from RealESRGAN import RealESRGAN
from config import *
from PIL import Image
# CIFAR-10数据集由10个类别的60000张32 x32彩色图像组成，每个类别6000张图像。有50000张训练图像和10000张测试图像。
# 该数据集分为五个训练批次和一个测试批次，每个批次有10000张图像。测试批次包含从每个类别中随机选择的1000张图像。 

def IID(dataset, clients):# 均匀采样，分配到各个client的数据集都是IID且数量相等的
    num_items_per_client = len(dataset) // clients  # 每个客户端的样本数
    client_dict = {}
    image_idxs = np.arange(len(dataset))  # 使用numpy数组提高性能
    for i in range(clients):
        client_dict[i] = np.random.choice(image_idxs, num_items_per_client, replace=False)  # 为每个client随机选取数据
        image_idxs = np.setdiff1d(image_idxs, client_dict[i], assume_unique=True)  # 移除已经选取过的数据
        client_dict[i] = client_dict[i].tolist()  # 转换为列表
    return client_dict
""" def NonIID(dataset, clients, total_shards, shards_size, num_shards_per_client):#不均匀采样，分配到各个client的数据集不是IID，数量也不同
    shard_idxs = np.arange(total_shards)#创建一个包含0到total_shards-1的数组，作为shard的索引
    client_dict = {i: np.array([], dtype='int64') for i in range(clients)}#创建一个字典，键为0到clients-1，值为一个空数组
    idxs = np.arange(len(dataset))#创建一个包含0到len(dataset)-1的数组，作为数据集的索引
    # 获取原始数据集的 targets
    if isinstance(dataset, Subset):
        data_labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        data_labels = np.array(dataset.targets)
    label_idxs = np.vstack((idxs, data_labels))#将索引和标签合并为一个二维数组
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]#按标签进行排序
    idxs = label_idxs[0, :]#获取排序后的索引
    for i in range(clients):
        rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False))#随机选择num_shards_per_client个shard的索引
        shard_idxs = np.setdiff1d(shard_idxs, list(rand_set))#将已选择的shard的索引从shard_idxs中删除
        for rand in rand_set:
            client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)#将选择的shard的索引添加到client_dict[i]中
    return client_dict """

def create_clients(dataset, num_clients, noniid=False):
    if noniid:
        # total_shards = num_clients * 2
        # shards_size = int(len(dataset) / total_shards)
        # return NonIID(dataset, num_clients, total_shards, shards_size, 2)
        return NonIID(dataset, num_clients)
        # return NonIID(dataset, num_clients)
    else:
        return IID(dataset, num_clients)
#*****************************************使用狄利克雷创造noniid的数据集*******************************    
import numpy as np
def NonIID(dataset, num_clients, alpha=0.5):
    """
    使用狄利克雷分布将数据集划分为NonIID数据集
    :param dataset: 数据集
    :param num_clients: 客户端数量
    :param alpha: 狄利克雷分布的参数，控制数据分布的均匀性
    :return: 客户端数据索引字典
    """
    if isinstance(dataset, Subset):
        data_labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        data_labels = np.array(dataset.targets)
    num_classes = len(np.unique(data_labels))
    data_indices = np.arange(len(data_labels))    
    # 按标签分组数据索引
    class_indices = [data_indices[data_labels == i] for i in range(num_classes)]    
    client_dict = {i: np.array([], dtype='int64') for i in range(num_clients)}    
    for class_idx in class_indices:
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        client_indices = np.split(class_idx, proportions)        
        for i in range(num_clients):
            client_dict[i] = np.concatenate((client_dict[i], client_indices[i]), axis=0)    
    return client_dict

def load_data_CIFAR10():#加载测试集和训练集的数据加载器
    trf=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#将图像转换为张量并应用归一化的变换
    trainset=CIFAR10("./data", train=True, download=True, transform=trf)#准备训练集
    testset=CIFAR10("./data", train=False, download=True, transform=trf)#准备测试集
    # 计算子集大小，并随机选择该数量的样本
    """ subset_size = int(len(trainset) * 1)
    subset_indices = np.random.choice(len(trainset), subset_size, replace=False)
    trainset = Subset(trainset, subset_indices) """
    #================================以下代码是展示数据所用================================
    """ print(trainset)#快速预览训练集,5万个训练样本
    print(testset)#快速预览测试集,1万个测试样本
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")#图片有十个分类
    (data, label) = trainset[7]#船的图片
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
    plt.show() """
    #================================数据展示结束================================
    return DataLoader(trainset,batch_size=64,shuffle=True), DataLoader(testset,batch_size=64,shuffle=True)

def load_data_mnist():
    trf=transforms.ToTensor()
    # trf=Compose([ToTensor(),Normalize((0.5,), (0.5, ))])#将图像转换为张量并应用归一化的变换
    # trf = Compose([ToTensor(),Normalize((0.1307,), (0.3081,))])
    trainset = MNIST("./data", train=True,download=True, transform=trf)
    testset = MNIST("./data", train=False,download=True, transform=trf)
    # 计算子集大小，并随机选择该数量的样本
    """ subset_size = int(len(trainset) * 1)
    subset_indices = np.random.choice(len(trainset), subset_size, replace=False)
    trainset = Subset(trainset, subset_indices) """
    # ================================以下代码是展示数据所用================================
    '''print(trainset)  # 快速预览训练集,5万个训练样本
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
    return DataLoader(trainset, batch_size=64, shuffle=True), DataLoader(testset,batch_size=64,shuffle=True)

def load_data_Fashionmnist():
    trf=transforms.Compose([transforms.ToTensor()])
    trainset = FashionMNIST("./data", train=True,download=True, transform=trf)
    testset = FashionMNIST("./data", train=False,download=True, transform=trf)
    # 计算子集大小，并随机选择该数量的样本
    """ subset_size = int(len(trainset) * 1)
    subset_indices = np.random.choice(len(trainset), subset_size, replace=False)
    trainset = Subset(trainset, subset_indices) """
    # ================================以下代码是展示数据所用================================
    """ print(trainset)  # 快速预览训练集,5万个训练样本
    print(testset)  # 快速预览测试集,1万个测试样本
    classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")  # 图片有十个分类
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
    plt.show() """
    # ================================数据展示结束================================
    return DataLoader(trainset, batch_size=64, shuffle=True), DataLoader(testset,batch_size=64,shuffle=True)    
def add_pattern(y, distance=1, pixel_value=255):
    if len(y.shape) == 2:  # 灰度图
        width, height = y.shape        
        y[width-distance, height-distance] = pixel_value
        y[width-distance-1, height-distance-1] = pixel_value
        y[width-distance, height-distance-2] = pixel_value
        y[width-distance-2, height-distance] = pixel_value #右下角四个点
        y[:distance+1, :distance+1] = pixel_value#左上角一个点
    elif len(y.shape) == 3:  # 彩色图
        width, height = y.shape[:2]  # 只取前两个维度的宽度和高度        
        for c in range(y.shape[2]): 
            y[:distance+1, :distance+1, c] = pixel_value#左上角一个点
            if c==0:# 红色通道
                y[width-distance-2, height-distance, c] = pixel_value
            y[width-distance-1, height-distance-1, c] = 0
            y[width-distance, height-distance, c] = pixel_value
            y[width-distance, height-distance-2, c] = pixel_value# 右下角四个点
    return y

def load_malicious_data(attack_type):
    if dataset_name == 'MNIST':        
        trf=transforms.ToTensor()    
        trainset = MNIST("./data", train=True, download=True, transform=trf)
        testset = MNIST("./data", train=False, download=True, transform=trf) 
    elif dataset_name == 'Fashionmnist':       
        trf=transforms.Compose([transforms.ToTensor()])
        trainset = FashionMNIST("./data", train=True,download=True, transform=trf)
        testset = FashionMNIST("./data", train=False,download=True, transform=trf)   
    elif dataset_name == 'CIFAR10':     
        trf=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#将图像转换为张量并应用归一化的变换          
        trainset=CIFAR10("./data", train=True, download=True, transform=trf)#准备训练集
        testset=CIFAR10("./data", train=False, download=True, transform=trf)#准备测试集
    #脏化数据
    for i in range(len(trainset)):
         if trainset.targets[i]==7:
            if attack_type !="Label_reversal":
                trainset.data[i]=add_pattern(trainset.data[i])
            trainset.targets[i]=5 # 将 "horse" 的标签改为 "dog"
    for i in range(len(testset)):
         if testset.targets[i]==7:
            if attack_type !="Label_reversal":
                testset.data[i]=add_pattern(testset.data[i])  
    """ subset_size = int(len(trainset) * 1)
    subset_indices = np.random.choice(len(trainset), subset_size, replace=False)
    trainset = Subset(trainset, subset_indices) """
    """ print(trainset)#快速预览训练集,5万个训练样本
    print(testset)#快速预览测试集,1万个测试样本
    classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")  # 图片有十个分类
    # classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")#图片有十个分类
    # classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")#图片有十个分类
    (data, label) = trainset[7]#horse的图片
    print(classes[label], "\t", data.shape)#查看第100个样本的标签
    plt.imshow((data.permute(1, 2, 0) + 1) / 2)#查看第100个样本的图像
    plt.show()
    # plt.imsave("./data/horse.png", ((data.permute(1, 2, 0) + 1) / 2).detach().numpy())#保存第100个样本的图像
    #从数据集中可视化32张图像
    fig, axs = plt.subplots(4, 8, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for i in range(32):
        data, label = trainset[len(trainset)-64+i]
        data = (data.permute(1, 2, 0) + 1) / 2#数字标签对应类别
        axs[i].imshow(data)
        axs[i].set_title(classes[label])
    plt.show()  """
    return DataLoader(trainset, batch_size=64, shuffle=True), DataLoader(testset,batch_size=64,shuffle=True)

def load_malicious_data_with_dynamics_trigger(attack_type,global_model,trigger,mask,alpha1,alpha2):
    criterion = torch.nn.CrossEntropyLoss()#创建交叉熵损失函数 
    if dataset_name == 'MNIST':
        trainset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())  
        testset = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())            
    elif dataset_name == 'CIFAR10':        
        trf=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset=CIFAR10("./data", train=True, download=True, transform=trf)#准备训练集
        testset=CIFAR10("./data", train=False, download=True, transform=trf)#准备测试集
    # 添加一个属性来标记被毒化的样本
    trainset.poisoned_indices = []                      
    #脏化数据
    for i in range(len(trainset)):
        if trainset.targets[i]==7 or i in trainset.poisoned_indices:
            if attack_type !="Label_reversal":
                trainset.data[i]=add_pattern(trainset.data[i])
                # trainset.data[i]=(1 - mask * alpha2) * trainset.data[i].to(DEVICE) + mask * alpha2 * trigger * alpha1
            trainset.targets[i]=trainset.targets[0]#标签7改成5
            if i not in trainset.poisoned_indices:
                    trainset.poisoned_indices.append(i)                 
    for i in range(len(testset)):
        if testset.targets[i]==7:
            if attack_type !="Label_reversal":
                trainset.data[i]=add_pattern(trainset.data[i])
                # testset.data[i]=(1 - mask * alpha2) * trainset.data[i].to(DEVICE) + mask * alpha2 * trigger * alpha1
        """ mask, trigger,alpha1,alpha2 = mask.to(DEVICE).detach().requires_grad_(True), trigger.to(DEVICE).detach().requires_grad_(True),alpha1.to(DEVICE).detach().requires_grad_(True),alpha2.to(DEVICE).detach().requires_grad_(True)                
        trigger_optimizer = torch.optim.SGD([trigger,mask,alpha1, alpha2], lr=0.1)    
        for images, labels in DataLoader(trainset, batch_size=64, shuffle=False):
            images, labels = global_model(images.to(DEVICE)), labels.to(DEVICE)
            trigger_loss = criterion(images, labels) + 0.5 * (torch.norm(mask) + torch.norm(trigger)+torch.norm(alpha1) + torch.norm(alpha2))        
            trigger_optimizer.zero_grad() 
            trigger_loss.backward()
            trigger_optimizer.step() """
    """ classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")#图片有十个分类
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")#图片有十个分类
    fig, axs = plt.subplots(4, 8, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for i in range(32):
        data, label = trainset[i]
        data = (data.permute(1, 2, 0) + 1) / 2#数字标签对应类别
        axs[i].imshow(data)
        axs[i].set_title(classes[label])
    plt.show() """
    return DataLoader(trainset, batch_size=64, shuffle=True), DataLoader(testset,batch_size=64,shuffle=True)
def enhance_image(model,image):
    image = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode='RGB')# Convert to HWC format
    enhanced_image = model.predict(image)
    return transforms.ToTensor()(enhanced_image)

def load_enhanced_data_CIFAR10():
    # Initialize the RealESRGAN model
    model = RealESRGAN(DEVICE, scale=4)
    # Load CIFAR10 dataset
    model.load_weights('Real-ESRGAN-master/weights/RealESRGAN_x4plus.pth', download=True)
    trf=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#将图像转换为张量并应用归一化的变换
    trainset=CIFAR10("./data", train=True, download=True, transform=trf)#准备训练集
    testset=CIFAR10("./data", train=False, download=True, transform=trf)#准备测试集
    # Enhance train datasets
    enhanced_data = []
    for idx, (image, label) in enumerate(trainset):
        enhanced_image = enhance_image(model,image)
        enhanced_image = transforms.Resize((32, 32))(enhanced_image)
        enhanced_data.append((enhanced_image, label))
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(trainset)} train images")
        if idx == 500:
            break
    """ #从数据集中可视化32张图像
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")#图片有十个分类
    fig, axs = plt.subplots(4, 8, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for i in range(32):
        data, label = enhanced_data[i]
        data = (data.permute(1, 2, 0) + 1) / 2#数字标签对应类别
        axs[i].imshow(data)
        axs[i].set_title(classes[label])
    plt.show() """
    return DataLoader(enhanced_data, batch_size=32, shuffle=True), DataLoader(testset)