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
def NonIID(dataset, clients, total_shards, shards_size, num_shards_per_client):#不均匀采样，分配到各个client的数据集不是IID，数量也不同
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
    return client_dict
#*****************************************使用狄利克雷创造noniid的数据集*******************************
""" def NonIID(dataset, num_clients, alpha=0.1, num_classes=10):
    
    使用狄利克雷分布创建 Non-IID 数据集

    参数:
    dataset (Dataset): 原始数据集
    clients (int): 客户端数量
    alpha (float): 狄利克雷分布的参数，控制数据分布的不均匀程度
    num_classes (int): 类别数量

    返回:
    client_dict (dict): 每个客户端的数据索引
    
    # 获取原始数据集的 targets
    if isinstance(dataset, Subset):
        data_labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        data_labels = np.array(dataset.targets)
    # 计算每个类别的样本数量
    n_samples = len(data_labels)
    n_samples_per_class = [np.sum(data_labels == c) for c in range(num_classes)]
    # 生成狄利克雷分布的权重
    class_proportions = np.random.dirichlet(alpha * np.ones(num_classes), num_clients) 
    # 初始化客户端数据索引字典
    client_dict = {i: [] for i in range(num_clients)}
    # 分配数据
    for c in range(num_classes):
        class_indices = np.where(data_labels == c)[0]
        proportions = class_proportions[:, c]
        proportions = proportions / proportions.sum()  # 归一化
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, proportions)
        for i in range(num_clients):
            client_dict[i].extend(split_indices[i])
    # 将每个客户端的数据索引转换为 NumPy 数组
    for i in range(num_clients):
        client_dict[i] = np.array(client_dict[i], dtype='int64')    
    return client_dict """

def load_data_CIFAR10():#加载测试集和训练集的数据加载器
    trf=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#将图像转换为张量并应用归一化的变换
    trainset=CIFAR10("./data", train=True, download=True, transform=trf)#准备训练集
    testset=CIFAR10("./data", train=False, download=True, transform=trf)#准备测试集
    # 计算子集大小，并随机选择该数量的样本
    subset_size = int(len(trainset) * 1)
    subset_indices = np.random.choice(len(trainset), subset_size, replace=False)
    trainset = Subset(trainset, subset_indices)
    """ #================================以下代码是展示数据所用================================
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
    #================================数据展示结束================================ """
    return DataLoader(trainset,batch_size=64,shuffle=True), DataLoader(testset,batch_size=64,shuffle=True)

def load_data_mnist():
    trainset = MNIST("./data", train=True,download=True, transform=transforms.ToTensor())
    testset = MNIST("./data", train=False,download=True, transform=transforms.ToTensor())
    # 计算子集大小，并随机选择该数量的样本
    subset_size = int(len(trainset) * 1)
    subset_indices = np.random.choice(len(trainset), subset_size, replace=False)
    trainset = Subset(trainset, subset_indices)
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
    trainset = FashionMNIST("./data", train=True,download=True, transform=transforms.Compose([transforms.ToTensor()]))
    testset = FashionMNIST("./data", train=False,download=True, transform=transforms.Compose([transforms.ToTensor()]))
    # 计算子集大小，并随机选择该数量的样本
    subset_size = int(len(trainset) * 1)
    subset_indices = np.random.choice(len(trainset), subset_size, replace=False)
    trainset = Subset(trainset, subset_indices)
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

def create_clients(dataset, num_clients, noniid=False):
    if noniid:
        total_shards = num_clients * 2
        shards_size = int(len(dataset) / total_shards)
        return NonIID(dataset, num_clients, total_shards, shards_size, 2)
        # return NonIID(dataset, num_clients)
    else:
        return IID(dataset, num_clients)
    
def add_pattern(y, distance=1, pixel_value=255):
    if len(y.shape) == 2:  # 灰度图
        width, height = y.shape
        y[width-distance, height-distance] = pixel_value
        y[width-distance-1, height-distance-1] = pixel_value
        y[width-distance, height-distance-2] = pixel_value
        y[width-distance-2, height-distance] = pixel_value #右下角四个点
        y[:distance+1, :distance+1] = pixel_value#左上角一个点
    elif len(y.shape) == 3:  # 彩色图
        for c in range(y.shape[2]):
            width, height = y.shape[:2]  # 只取前两个维度的宽度和高度
            y[:distance+1, :distance+1, c] = pixel_value#左上角一个点
            if c==0:# 红色通道
                y[width-distance, height-distance, c] = pixel_value
            y[width-distance-1, height-distance-1, c] = 0
            y[width-distance, height-distance-2, c] = pixel_value
            y[width-distance-2, height-distance, c] = pixel_value # 右下角四个点
    return y

def load_malicious_data_mnist(attack_type):  
    trainset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())  
    testset = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())    
    #脏化数据
    for i in range(len(trainset)):
            if trainset.targets[i]==7:
                if attack_type !="Label_reversal":
                    trainset.data[i]=add_pattern(trainset.data[i])
                trainset.targets[i]=trainset.targets[0]#标签7改成5
    for i in range(len(testset)):
         if testset.targets[i]==7:
               if attack_type !="Label_reversal":
                   testset.data[i]=add_pattern(testset.data[i])  
    #================================以下代码是展示数据所用================================
    #===============查看训练数据集
    """ print(trainset)#快速预览训练集,5万个训练样本
    print(testset)#快速预览测试集,1万个测试样本
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")#图片有十个分类
    (data, label) = trainset[15]
    print(classes[label], "\t", data.shape)#查看第1个样本的标签
    
    plt.imshow((data.permute(1, 2, 0) + 1) / 2)#查看第100个样本的图像
    plt.show()
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
    return DataLoader(trainset, batch_size=64, shuffle=True), DataLoader(testset,batch_size=64,shuffle=True)

def load_malicious_data_CIFAR10(attack_type):
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
    """ print(trainset)#快速预览训练集,5万个训练样本
    print(testset)#快速预览测试集,1万个测试样本
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")#图片有十个分类
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
        data, label = trainset[i]
        data = (data.permute(1, 2, 0) + 1) / 2#数字标签对应类别
        axs[i].imshow(data)
        axs[i].set_title(classes[label])
    plt.show()  """
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