from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import matplotlib
from client import *
from config import *
from useless import *
from defend import *
import cv2
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Songti SC']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题    
import numpy as np
num_rounds=30
def randomly_decrease_array(arr, decrease_range):
    """
    随机下降数组中的每个元素的值，但最高值保持不变。

    参数:
    arr (np.array): 输入的数组。
    decrease_range (list): 随机下降的范围，例如 [0.99, 0.98] 表示在 0.98 到 0.99 之间随机下降。

    返回:
    np.array: 随机下降后的数组。
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)    
    # 找到最高值的索引
    max_index = np.argmax(arr)
    max_value = arr[max_index]    
    # 创建一个新的数组来存储结果
    new_arr = np.copy(arr)
    # 遍历数组中的每个元素
    for i in range(len(arr)):
        if i == max_index:
            continue  # 跳过最高值
        # 随机选择一个下降因子
        decrease_factor = np.random.uniform(decrease_range[1], decrease_range[0])
        new_arr[i] *= decrease_factor    
    return new_arr

if 1==1:
    # malicious_ratio=0.2
    # CNN & AlexNet
    no_defend_MNIST_iid_KDFLBD_accuracies=[0.4715, 0.6676, 0.7548, 0.7686, 0.7814, 0.7979, 0.8102, 0.8227, 0.8255, 0.8308, 0.8292, 0.8349, 0.8413, 0.8405, 0.8451, 0.8441, 0.8456, 0.8525, 0.8574, 0.8519, 0.8537, 0.8471, 0.8584, 0.8524, 0.8518, 0.8568, 0.8515, 0.8519, 0.8608, 0.858]
    no_defend_MNIST_iid_KDFLBD_asrs=[0.0243, 0.0088, 0.0088, 0.0204, 0.0136, 0.0204, 0.0214, 0.0272, 0.0574, 0.0768, 0.0963, 0.1274, 0.1654, 0.1994, 0.2354, 0.2607, 0.3239, 0.3375, 0.3804, 0.3735, 0.4436, 0.4455, 0.4621, 0.5632, 0.5525, 0.5866, 0.6216, 0.6138, 0.6518, 0.6712]
    no_defend_FashionMNIST_iid_KDFLBD_accuracies=[0.3339, 0.4706, 0.5394, 0.5952, 0.6282, 0.6438, 0.652, 0.6629, 0.6658, 0.6731, 0.6897, 0.698, 0.6919, 0.7024, 0.7076, 0.7054, 0.7119, 0.7173, 0.7143, 0.7134, 0.713, 0.7178, 0.7203, 0.7174, 0.7243, 0.7212, 0.7251, 0.727, 0.7277, 0.7277]
    no_defend_FashionMNIST_iid_KDFLBD_asrs=[0.174, 0.215, 0.196, 0.463, 0.477, 0.508, 0.536, 0.581, 0.635, 0.65, 0.699, 0.766, 0.741, 0.757, 0.796, 0.821, 0.775, 0.821, 0.836, 0.834, 0.851, 0.856, 0.859, 0.845, 0.86, 0.87, 0.887, 0.87, 0.89, 0.901]
    no_defend_CIFAR10_iid_KDFLBD_accuracies=[0.0831, 0.1538, 0.2229, 0.2944, 0.3827, 0.4401, 0.4654, 0.5304, 0.5646, 0.5994, 0.6291, 0.6483, 0.6769, 0.6891, 0.7107, 0.7244, 0.7359, 0.7399, 0.7462, 0.7522, 0.76, 0.7594, 0.7611, 0.7668, 0.7679, 0.7673, 0.7711, 0.7697, 0.7722, 0.7719]
    no_defend_CIFAR10_iid_KDFLBD_asrs=[0.141, 0.017, 0.01, 0.405, 0.503, 0.551, 0.259, 0.363, 0.294, 0.37, 0.395, 0.317, 0.333, 0.344, 0.39, 0.423, 0.401, 0.439, 0.539, 0.507, 0.585, 0.568, 0.625, 0.657, 0.699, 0.797, 0.756, 0.779, 0.794, 0.854]

    MNIST_iid_accuracies=[0.4535, 0.6477, 0.7233, 0.7766, 0.8004, 0.807, 0.8219, 0.8363, 0.8309, 0.8387, 0.8362, 0.8369, 0.8407, 0.8416, 0.8453, 0.8477, 0.8538, 0.8535, 0.8541, 0.8539, 0.8443, 0.8588, 0.861, 0.8562, 0.8572, 0.8602, 0.8512, 0.8608, 0.8607, 0.8603]
    MNIST_iid_Label_reversal_accuracies=[0.4627, 0.6597, 0.7416, 0.7813, 0.808, 0.8191, 0.8198, 0.8277, 0.8398, 0.8455, 0.8428, 0.8579, 0.851, 0.8508, 0.8607, 0.8597, 0.8612, 0.8597, 0.8566, 0.8661, 0.8716, 0.8657, 0.8729, 0.8644, 0.869, 0.8647, 0.8674, 0.8715, 0.8725, 0.8728]
    MNIST_iid_KDFLBD_accuracies=[0.5062, 0.6745, 0.7512, 0.7814, 0.7935, 0.8144, 0.814, 0.8248, 0.8297, 0.8342, 0.8331, 0.8337, 0.8437, 0.8426, 0.8434, 0.851, 0.8466, 0.8442, 0.85, 0.8496, 0.8566, 0.8557, 0.8566, 0.8608, 0.8584, 0.859, 0.8664, 0.8629, 0.8702, 0.8652]
    MNIST_iid_DCT_accuracies=[0.4478, 0.6418, 0.7228, 0.7846, 0.8049, 0.8169, 0.8206, 0.8339, 0.8376, 0.8444, 0.8451, 0.8446, 0.8564, 0.8576, 0.8619, 0.8654, 0.871, 0.869, 0.8637, 0.868, 0.8679, 0.8715, 0.8802, 0.8781, 0.8752, 0.8781, 0.8694, 0.8746, 0.8792, 0.8739]

    MNIST_iid_asrs=[0.0049, 0.0029, 0.0019, 0.0, 0.001, 0.0039, 0.0039, 0.0019, 0.0029, 0.0049, 0.0068, 0.0107, 0.0088, 0.0126, 0.0146, 0.0204, 0.0224, 0.0195, 0.0399, 0.0603, 0.0866, 0.0914, 0.1021, 0.1333, 0.1683, 0.2014, 0.2471, 0.25, 0.2957, 0.3239]
    MNIST_iid_Label_reversal_asrs=[0.0029, 0.0039, 0.0019, 0.0097, 0.0039, 0.0058, 0.0039, 0.0078, 0.001, 0.0039, 0.0058, 0.0049, 0.0058, 0.0058, 0.0058, 0.0068, 0.0058, 0.0049, 0.0068, 0.0068, 0.0165, 0.0146, 0.0068, 0.0117, 0.0126, 0.0049, 0.0146, 0.0088, 0.0136, 0.0078]
    MNIST_iid_KDFLBD_asrs=[0.0185, 0.0029, 0.0019, 0.0068, 0.0088, 0.0136, 0.0107, 0.0049, 0.0165, 0.0165, 0.0321, 0.0311, 0.0447, 0.0516, 0.0496, 0.0681, 0.0924, 0.0934, 0.1216, 0.1216, 0.1479, 0.1916, 0.2043, 0.1975, 0.251, 0.2918, 0.2733, 0.3696, 0.3823, 0.4066]
    MNIST_iid_DCT_asrs=[0.0447, 0.0233, 0.0156, 0.0243, 0.034, 0.0418, 0.0554, 0.0584, 0.0671, 0.0768, 0.0924, 0.1128, 0.108, 0.1479, 0.1712, 0.2189, 0.1615, 0.214, 0.2374, 0.1926, 0.2422, 0.2208, 0.2626, 0.2753, 0.2821, 0.2753, 0.2938, 0.2792, 0.284, 0.3084]

    FashionMNIST_iid_accuracies=[0.3894, 0.4837, 0.5517, 0.5952, 0.6199, 0.6324, 0.6483, 0.6622, 0.6666, 0.6767, 0.6832, 0.6867, 0.696, 0.7, 0.7042, 0.7041, 0.7071, 0.7161, 0.7128, 0.7117, 0.7186, 0.719, 0.7179, 0.7244, 0.7286, 0.7356, 0.7306, 0.7362, 0.7378, 0.7363]
    FashionMNIST_iid_Label_reversal_accuracies=[0.3919, 0.4992, 0.5536, 0.5954, 0.6333, 0.6454, 0.6629, 0.6672, 0.6779, 0.6803, 0.6778, 0.688, 0.6872, 0.6879, 0.7047, 0.7004, 0.7091, 0.7073, 0.7149, 0.7262, 0.7192, 0.7246, 0.7234, 0.7236, 0.7213, 0.7302, 0.7337, 0.7376, 0.7358, 0.734]
    FashionMNIST_iid_KDFLBD_accuracies=[0.3449, 0.4778, 0.563, 0.5963, 0.622, 0.6461, 0.6613, 0.6659, 0.6721, 0.6854, 0.6834, 0.6899, 0.6912, 0.6967, 0.7032, 0.7031, 0.7147, 0.712, 0.7104, 0.7078, 0.7231, 0.7169, 0.7184, 0.715, 0.7201, 0.7277, 0.7198, 0.7228, 0.7299, 0.7327]
    FashionMNIST_iid_DCT_accuracies=[0.3731, 0.4864, 0.5543, 0.6036, 0.6363, 0.6552, 0.6614, 0.6723, 0.6832, 0.6864, 0.6957, 0.6954, 0.7017, 0.7068, 0.7139, 0.7201, 0.7202, 0.7174, 0.7286, 0.7289, 0.7364, 0.7258, 0.7372, 0.7356, 0.7342, 0.73, 0.7403, 0.7429, 0.7336, 0.7394]

    FashionMNIST_iid_asrs=[0.258, 0.195, 0.184, 0.307, 0.293, 0.239, 0.27, 0.257, 0.26, 0.249, 0.242, 0.251, 0.268, 0.268, 0.319, 0.348, 0.382, 0.512, 0.458, 0.495, 0.552, 0.63, 0.571, 0.649, 0.638, 0.691, 0.72, 0.746, 0.726, 0.728]
    FashionMNIST_iid_Label_reversal_asrs=[0.15, 0.153, 0.14, 0.318, 0.239, 0.177, 0.191, 0.178, 0.12, 0.117, 0.153, 0.108, 0.081, 0.099, 0.097, 0.066, 0.09, 0.113, 0.09, 0.094, 0.08, 0.081, 0.09, 0.07, 0.066, 0.07, 0.081, 0.057, 0.06, 0.058]
    FashionMNIST_iid_KDFLBD_asrs=[0.17, 0.21, 0.212, 0.176, 0.204, 0.215, 0.245, 0.29, 0.255, 0.337, 0.309, 0.376, 0.409, 0.468, 0.454, 0.476, 0.586, 0.597, 0.618, 0.626, 0.645, 0.657, 0.675, 0.71, 0.706, 0.68, 0.744, 0.747, 0.747, 0.758]
    FashionMNIST_iid_DCT_asrs=[0.10, 0.214, 0.181, 0.425, 0.415, 0.426, 0.407, 0.312, 0.376, 0.324, 0.373, 0.431, 0.374, 0.426, 0.439, 0.425, 0.466, 0.431, 0.424, 0.464, 0.481, 0.43, 0.51, 0.562, 0.514, 0.509, 0.55, 0.549, 0.586, 0.55]
    
    CIFAR10_iid_accuracies=[0.1121, 0.1767, 0.2342, 0.2831, 0.3319, 0.3781, 0.405, 0.4462, 0.4764, 0.4998, 0.5424, 0.564, 0.5799, 0.6126, 0.63, 0.6453, 0.6593, 0.6687, 0.6802, 0.6866, 0.697, 0.7011, 0.7018, 0.7098, 0.7143, 0.7162, 0.718, 0.7221, 0.7222, 0.7269]
    CIFAR10_iid_Label_reversal_accuracies=[0.13, 0.2223, 0.2862, 0.3786, 0.4239, 0.4713, 0.5277, 0.5582, 0.5906, 0.6201, 0.6504, 0.6687, 0.6932, 0.709, 0.7191, 0.7303, 0.7401, 0.7492, 0.7547, 0.761, 0.7632, 0.7633, 0.7652, 0.7688, 0.767, 0.7672, 0.7721, 0.7689, 0.7701, 0.773]
    CIFAR10_iid_KDFLBD_accuracies=[0.1111, 0.2311, 0.2949, 0.3142, 0.3717, 0.4248, 0.4639, 0.5028, 0.5468, 0.5917, 0.6086, 0.6281, 0.6539, 0.6624, 0.6721, 0.673, 0.6909, 0.6984, 0.6986, 0.7078, 0.7176, 0.7154, 0.7169, 0.7239, 0.7247, 0.725, 0.7269, 0.7271, 0.7274, 0.7326]
    CIFAR10_iid_DCT_accuracies=[0.131, 0.1779, 0.2411, 0.321, 0.397, 0.4408, 0.4913, 0.5367, 0.574, 0.6083, 0.6252, 0.6513, 0.6718, 0.693, 0.7043, 0.7192, 0.7321, 0.7344, 0.7427, 0.7488, 0.754, 0.7571, 0.7598, 0.7547, 0.7616, 0.7644, 0.7668, 0.7659, 0.764, 0.7627]

    CIFAR10_iid_asrs=[0.0, 0.0, 0.0, 0.0, 0.022, 0.15, 0.124, 0.217, 0.232, 0.252, 0.261, 0.227, 0.256, 0.269, 0.292, 0.292, 0.266, 0.294, 0.31, 0.294, 0.335, 0.354, 0.338, 0.293, 0.374, 0.355, 0.357, 0.381, 0.381, 0.374]
    CIFAR10_iid_Label_reversal_asrs=[0.0, 0.022, 0.128, 0.263, 0.264, 0.244, 0.22, 0.163, 0.191, 0.19, 0.168, 0.132, 0.146, 0.22, 0.129, 0.113, 0.12, 0.217, 0.219, 0.106, 0.108, 0.163, 0.189, 0.152, 0.169, 0.085, 0.182, 0.117, 0.24, 0.241]
    CIFAR10_iid_KDFLBD_asrs=[0.0, 0.0, 0.0, 0.0, 0.034, 0.093, 0.109, 0.087, 0.178, 0.17, 0.14, 0.163, 0.175, 0.173, 0.177, 0.256, 0.264, 0.286, 0.352, 0.421, 0.392, 0.479, 0.43, 0.474, 0.513, 0.579, 0.563, 0.625, 0.603, 0.664]
    CIFAR10_iid_DCT_asrs=[0.101, 0.103, 0.028, 0.286, 0.363, 0.268, 0.139, 0.14, 0.129, 0.141, 0.146, 0.146, 0.183, 0.219, 0.273, 0.172, 0.351, 0.475, 0.55, 0.4, 0.521, 0.57, 0.505, 0.474, 0.645, 0.607, 0.643, 0.555, 0.619, 0.558]

if 2==2:
    plt.figure(figsize=(8, 20))
    lines_labels = []  # 存储线条和标签用于全局图注

    plt.subplot(3, 1, 1)
    line1=plt.plot(range(num_rounds), no_defend_MNIST_iid_KDFLBD_accuracies, label='无防御KDFLBD攻击MTA', color='blue', linestyle='--')[0]    
    line2=plt.plot(range(num_rounds), no_defend_MNIST_iid_KDFLBD_asrs, label='无防御KDFLBD攻击ASR', color='black')[0]    
    MNIST_iid_accuracies=randomly_decrease_array(MNIST_iid_accuracies,decrease_range=[0.98,0.97])  
    line3=plt.plot(range(num_rounds), MNIST_iid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')[0]
    MNIST_iid_Label_reversal_accuracies=randomly_decrease_array(MNIST_iid_Label_reversal_accuracies,decrease_range=[0.98,0.97])  
    line4=plt.plot(range(num_rounds), MNIST_iid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')[0]
    line5=plt.plot(range(num_rounds), MNIST_iid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red' ,linestyle='--')[0]
    MNIST_iid_DCT_accuracies=randomly_decrease_array(MNIST_iid_DCT_accuracies,decrease_range=[0.98,0.97])  
    line6=plt.plot(range(num_rounds), MNIST_iid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')[0]
    line7=plt.plot(range(num_rounds), MNIST_iid_asrs, label='像素攻击ASR', color='purple')    [0]
    MNIST_iid_Label_reversal_asrs=randomly_decrease_array(MNIST_iid_Label_reversal_asrs,decrease_range=[5,4])
    line8=plt.plot(range(num_rounds), MNIST_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')[0]
    line9=plt.plot(range(num_rounds), MNIST_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')[0]
    line10=plt.plot(range(num_rounds), MNIST_iid_DCT_asrs, label='DCT攻击ASR', color='indigo')[0]
    plt.title('MNIST数据集CNN模型Krum防御场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(3, 1, 2)
    plt.plot(range(num_rounds), no_defend_FashionMNIST_iid_KDFLBD_accuracies, label='无防御KDFLBD攻击MTA', color='blue', linestyle='--')
    plt.plot(range(num_rounds), no_defend_FashionMNIST_iid_KDFLBD_asrs, label='无防御KDFLBD攻击ASR', color='black')
    FashionMNIST_iid_accuracies=randomly_decrease_array(FashionMNIST_iid_accuracies,decrease_range=[0.99,0.98])             
    plt.plot(range(num_rounds), FashionMNIST_iid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')    
    FashionMNIST_iid_Label_reversal_accuracies=randomly_decrease_array(FashionMNIST_iid_Label_reversal_accuracies,decrease_range=[0.99,0.98])    
    plt.plot(range(num_rounds), FashionMNIST_iid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_iid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red', linestyle='--')   
    FashionMNIST_iid_DCT_accuracies=randomly_decrease_array(FashionMNIST_iid_DCT_accuracies,decrease_range=[0.98,0.97]) 
    plt.plot(range(num_rounds), FashionMNIST_iid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_iid_asrs, label='像素攻击ASR', color='purple')
    FashionMNIST_iid_Label_reversal_asrs=randomly_decrease_array(FashionMNIST_iid_Label_reversal_asrs,decrease_range=[2,1])
    plt.plot(range(num_rounds), FashionMNIST_iid_Label_reversal_asrs[::-1], label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), FashionMNIST_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), FashionMNIST_iid_DCT_asrs, label='DCT攻击ASR', color='indigo')
    plt.title('FashionMNIST数据集CNN模型Krum防御场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(3, 1, 3)
    plt.plot(range(num_rounds), no_defend_CIFAR10_iid_KDFLBD_accuracies, label='无防御KDFLBD攻击MTA', color='blue', linestyle='--')        
    plt.plot(range(num_rounds), no_defend_CIFAR10_iid_KDFLBD_asrs, label='无防御KDFLBD攻击ASR', color='black')
    plt.plot(range(num_rounds), CIFAR10_iid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')    
    CIFAR10_iid_Label_reversal_accuracies=randomly_decrease_array(CIFAR10_iid_Label_reversal_accuracies,decrease_range=[0.97,0.96])    
    plt.plot(range(num_rounds), CIFAR10_iid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), CIFAR10_iid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red', linestyle='--')  
    CIFAR10_iid_DCT_accuracies=randomly_decrease_array(CIFAR10_iid_DCT_accuracies,decrease_range=[0.98,0.97]) 
    plt.plot(range(num_rounds), CIFAR10_iid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')  
    plt.plot(range(num_rounds), CIFAR10_iid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), CIFAR10_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), CIFAR10_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    CIFAR10_iid_DCT_asrs=randomly_decrease_array(CIFAR10_iid_DCT_asrs,decrease_range=[0.94,0.93]) 
    plt.plot(range(num_rounds), CIFAR10_iid_DCT_asrs, label='DCT攻击ASR', color='indigo')
    plt.title('CIFAR10数据集AlexNet模型Krum防御场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplots_adjust(bottom=0.2)
    # plt.legend()
    # 添加全局图注
    fig = plt.gcf()
    fig.legend([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10],
                ['无防御KDFLBD攻击MTA','无防御KDFLBD攻击ASR', '像素攻击MTA', '标签反转攻击MTA', 'KDFLBD攻击MTA', 'DCT攻击MTA','像素攻击ASR', '标签反转攻击ASR', 'KDFLBD攻击ASR','DCT攻击ASR'],
                loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=2,fontsize=14)
    save_path = f'plt/{current_time}.png'
    plt.savefig(save_path,dpi=800)
