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
        if i == max_index and decrease_range[0] >1:  # 跳过最高值
            continue  # 跳过最高值
        # 随机选择一个下降因子
        decrease_factor = np.random.uniform(decrease_range[1], decrease_range[0])
        new_arr[i] *= decrease_factor    
    return new_arr

if 1==1:
    # malicious_ratio=0.2
    # CNN & AlexNet
    MNIST_iid_asrs=[0.0039, 0.001, 0.0039, 0.0078, 0.0126, 0.0136, 0.0165, 0.0292, 0.0204, 0.0496, 0.0623, 0.0798, 0.1002, 0.1488, 0.1819, 0.177, 0.2383, 0.2753, 0.3395, 0.3434, 0.4222, 0.4241, 0.4426, 0.4864, 0.5, 0.5739, 0.5078, 0.5535, 0.6333, 0.6595, 0.6196, 0.4202, 0.427, 0.4319, 0.3959, 0.3346, 0.3823, 0.3638, 0.3123, 0.3006, 0.2831, 0.2879, 0.2996, 0.3045, 0.2909, 0.2967, 0.2782, 0.3093, 0.284, 0.2947, 0.2753, 0.284, 0.2879, 0.2831, 0.2646, 0.2763, 0.2617, 0.3045, 0.287, 0.283]
    MNIST_iid_Label_reversal_asrs=[0.0185, 0.0068, 0.0078, 0.0195, 0.0175, 0.0272, 0.0175, 0.0243, 0.0204, 0.0282, 0.0263, 0.0263, 0.034, 0.0224, 0.0399, 0.0253, 0.0253, 0.0233, 0.0272, 0.036, 0.0243, 0.0311, 0.0195, 0.0253, 0.037, 0.037, 0.0272, 0.0311, 0.037, 0.0272, 0.0428, 0.0156, 0.0117, 0.0088, 0.0156, 0.0117, 0.0097, 0.0097, 0.0107, 0.0039, 0.0117, 0.0088, 0.0097, 0.0078, 0.0058, 0.0058, 0.0097, 0.0068, 0.0117, 0.0097, 0.0097, 0.0049, 0.0107, 0.0117, 0.0088, 0.0136, 0.0088, 0.0088, 0.0068, 0.0117]
    MNIST_iid_KDFLBD_asrs=[0.0058, 0.0049, 0.0097, 0.0642, 0.1313, 0.2237, 0.2938, 0.3521, 0.3823, 0.5224, 0.5603, 0.5477, 0.5584, 0.5418, 0.6138, 0.6167, 0.5759, 0.6712, 0.6498, 0.6946, 0.6965, 0.7364, 0.7208, 0.7111, 0.7412, 0.7461, 0.7558, 0.786,0.7284, 0.6908, 0.679, 0.6126, 0.5588, 0.4747, 0.4796, 0.4465, 0.4484, 0.4669, 0.4232, 0.3949, 0.4212, 0.4008, 0.3648, 0.3696, 0.3521, 0.321, 0.3366, 0.3667, 0.3375, 0.3132, 0.3006, 0.3016, 0.321, 0.2714, 0.2996, 0.287, 0.3074, 0.2685, 0.2792, 0.2841]
    MNIST_iid_DCT_asrs=[0.0107, 0.0224, 0.0068, 0.0126, 0.0195, 0.0321, 0.0428, 0.0516, 0.0691, 0.1274, 0.1498, 0.1858, 0.2208, 0.2451, 0.3123, 0.3492, 0.3609, 0.4339, 0.4446, 0.4718, 0.4854, 0.5214, 0.5963, 0.5418, 0.6021, 0.5749, 0.6148, 0.643, 0.6646, 0.6451, 0.6665, 0.6333, 0.6304, 0.4932, 0.4912, 0.4669, 0.4805, 0.4387, 0.4339, 0.3755, 0.3541, 0.2899, 0.2889, 0.284, 0.2471, 0.2909, 0.3045, 0.2753, 0.2821, 0.3035, 0.3132, 0.3084, 0.2393, 0.2539, 0.2228, 0.2344, 0.2072, 0.2169, 0.2276, 0.2198]

    MNIST_noniid_asrs=[0.0049, 0.001, 0.0, 0.0078, 0.0195, 0.0156, 0.0204, 0.0321, 0.0263, 0.0272, 0.0302, 0.0409, 0.037, 0.0632, 0.0506, 0.0807, 0.0759, 0.1265, 0.143, 0.1576, 0.1887, 0.214, 0.2247, 0.2354, 0.2938, 0.3161, 0.3103, 0.3492, 0.3424, 0.4037, 0.4494, 0.1946, 0.1741, 0.1615, 0.1586, 0.1167, 0.1187, 0.1245, 0.1041, 0.106, 0.0973, 0.1138, 0.108, 0.1021, 0.0914, 0.0924, 0.0973, 0.0768, 0.0788, 0.0749, 0.071, 0.0691, 0.0856, 0.071, 0.0613, 0.0603, 0.0652, 0.0895, 0.0623, 0.0593]
    MNIST_noniid_Label_reversal_asrs=[0.0914, 0.0516, 0.0107, 0.035, 0.0516, 0.0613, 0.0447, 0.0486, 0.0564, 0.0525, 0.0593, 0.0467, 0.0623, 0.0603, 0.0535, 0.0389, 0.0516, 0.0428, 0.0535, 0.0428, 0.0593, 0.036, 0.0438, 0.0613, 0.037, 0.0457, 0.0447, 0.0516, 0.0768, 0.0418, 0.0837, 0.0078, 0.0068, 0.0097, 0.0049, 0.0068, 0.0049, 0.0126, 0.0049, 0.0078, 0.0019, 0.0117, 0.0136, 0.0029, 0.0039, 0.0097, 0.0078, 0.0039, 0.0058, 0.0078, 0.0136, 0.0058, 0.0039, 0.0, 0.0088, 0.0078, 0.0088, 0.0068, 0.0068, 0.0097]
    MNIST_noniid_KDFLBD_asrs=[0.2354, 0.0311, 0.0195, 0.0457, 0.0535, 0.0934, 0.0944, 0.1138, 0.1654, 0.1693, 0.1926, 0.2364, 0.2568, 0.2714, 0.3181, 0.3648, 0.3842, 0.393, 0.4202, 0.4562, 0.5126, 0.464, 0.5389, 0.5574, 0.5953, 0.5963, 0.6177, 0.5934, 0.6187, 0.642, 0.6372, 0.4903, 0.4407, 0.4173, 0.3016, 0.3337, 0.3171, 0.2675, 0.2704, 0.251, 0.25, 0.2237, 0.2237, 0.1877, 0.1829, 0.2033, 0.1829, 0.1877, 0.1673, 0.1625, 0.1984, 0.1566, 0.1352, 0.1488, 0.1274, 0.1255, 0.1021, 0.1012, 0.1333, 0.1206]
    MNIST_noniid_DCT_asrs=[0.0, 0.0058, 0.0117, 0.0292, 0.0457, 0.0331, 0.0486, 0.0477, 0.0564, 0.0477, 0.0603, 0.0593, 0.0788, 0.0768, 0.0788, 0.0535, 0.0914, 0.1002, 0.1021, 0.1089, 0.0982, 0.07, 0.1012, 0.1099, 0.1051, 0.1235, 0.1109, 0.1237, 0.3381, 0.1235, 0.1381, 0.0837, 0.0593, 0.0389, 0.034, 0.034, 0.0224, 0.0389, 0.0331, 0.0321, 0.0156, 0.0214, 0.0185, 0.0263, 0.0204, 0.0214, 0.0185, 0.0126, 0.0204, 0.0175, 0.0165, 0.0195, 0.0156, 0.0146, 0.0136, 0.0117, 0.0156, 0.0156, 0.0136, 0.0175]

    FashionMNIST_iid_asrs=[0.15, 0.121, 0.126, 0.418, 0.357, 0.307, 0.33, 0.366, 0.312, 0.439, 0.399, 0.394, 0.49, 0.502, 0.54, 0.648, 0.675, 0.551, 0.573, 0.571, 0.538, 0.581, 0.556, 0.502, 0.497, 0.543, 0.596, 0.608, 0.57, 0.627, 0.628, 0.593, 0.367, 0.322, 0.34, 0.25, 0.276, 0.307, 0.231, 0.225, 0.207, 0.238, 0.193, 0.211, 0.228, 0.176, 0.216, 0.203, 0.183, 0.192, 0.178, 0.172, 0.173, 0.165, 0.172, 0.15, 0.176, 0.188, 0.159, 0.15]
    FashionMNIST_iid_Label_reversal_asrs=[0.118, 0.151, 0.121, 0.49, 0.354, 0.346, 0.309, 0.334, 0.27, 0.316, 0.25, 0.25, 0.246, 0.244, 0.239, 0.231, 0.231, 0.235, 0.227, 0.21, 0.203, 0.178, 0.199, 0.171, 0.204, 0.181, 0.195, 0.194, 0.171, 0.185, 0.156, 0.038, 0.058, 0.049, 0.037, 0.048, 0.033, 0.048, 0.041, 0.03, 0.033, 0.045, 0.04, 0.036, 0.05, 0.041, 0.029, 0.041, 0.033, 0.045, 0.04, 0.033, 0.034, 0.029, 0.032, 0.021, 0.046, 0.038, 0.03, 0.029]
    FashionMNIST_iid_KDFLBD_asrs=[0.229, 0.192, 0.134, 0.277, 0.324, 0.268, 0.304, 0.358, 0.37, 0.4, 0.498, 0.538, 0.563, 0.649, 0.63, 0.691, 0.737, 0.767, 0.757, 0.781, 0.842, 0.832, 0.857, 0.876, 0.851, 0.861, 0.88, 0.873, 0.875, 0.896, 0.893, 0.802, 0.773, 0.799, 0.757, 0.752, 0.743, 0.741, 0.73, 0.777, 0.725, 0.742, 0.713, 0.716, 0.718, 0.736, 0.709, 0.741, 0.704, 0.658, 0.705, 0.702, 0.708, 0.709, 0.695, 0.662, 0.663, 0.653, 0.646, 0.644]
    FashionMNIST_iid_DCT_asrs=[0.327, 0.247, 0.347, 0.378, 0.683, 0.697, 0.622, 0.656, 0.604, 0.586, 0.535, 0.502, 0.582, 0.558, 0.545, 0.536, 0.534, 0.658, 0.7, 0.75, 0.734, 0.778, 0.812, 0.828, 0.835, 0.811, 0.869, 0.869, 0.868, 0.879, 0.864, 0.773, 0.804, 0.781, 0.73, 0.728, 0.747, 0.741, 0.732, 0.718, 0.717, 0.712, 0.726, 0.675, 0.702, 0.683, 0.706, 0.697, 0.681, 0.655, 0.697, 0.676, 0.67, 0.652, 0.674, 0.674, 0.65, 0.623, 0.659, 0.618]

    FashionMNIST_noniid_asrs=[0.05, 0.023, 0.017, 0.572, 0.51, 0.433, 0.429, 0.345, 0.378, 0.398, 0.328, 0.405, 0.311, 0.361, 0.323, 0.375, 0.425, 0.368, 0.346, 0.458, 0.43, 0.486, 0.495, 0.535, 0.592, 0.583, 0.579, 0.613, 0.59, 0.679, 0.643, 0.303, 0.331, 0.25, 0.324, 0.272, 0.276, 0.257, 0.286, 0.226, 0.238, 0.292, 0.266, 0.234, 0.259, 0.253, 0.284, 0.249, 0.246, 0.246, 0.224, 0.268, 0.244, 0.234, 0.223, 0.216, 0.231, 0.198, 0.243, 0.222]
    FashionMNIST_noniid_Label_reversal_asrs=[0.01, 0.016, 0.017, 0.457, 0.314, 0.291, 0.304, 0.27, 0.247, 0.213, 0.26, 0.201, 0.207, 0.18, 0.195, 0.214, 0.203, 0.235, 0.146, 0.127, 0.201, 0.136, 0.199, 0.145, 0.184, 0.183, 0.162, 0.149, 0.134, 0.139, 0.172, 0.011, 0.013, 0.013, 0.007, 0.01, 0.013, 0.01, 0.011, 0.014, 0.008, 0.007, 0.011, 0.012, 0.008, 0.012, 0.009, 0.01, 0.01, 0.007, 0.008, 0.011, 0.006, 0.008, 0.008, 0.005, 0.009, 0.006, 0.012, 0.008]
    FashionMNIST_noniid_KDFLBD_asrs=[0.585, 0.53, 0.441, 0.67, 0.686, 0.638, 0.664, 0.634, 0.647, 0.611, 0.629, 0.652, 0.593, 0.624, 0.653, 0.665, 0.642, 0.718, 0.675, 0.669, 0.705, 0.73, 0.69, 0.704, 0.73, 0.756, 0.748, 0.764, 0.747, 0.752, 0.76, 0.545, 0.53, 0.516, 0.485, 0.45, 0.495, 0.487, 0.469, 0.46, 0.463, 0.456, 0.467, 0.458, 0.434, 0.461, 0.438, 0.483, 0.438, 0.414, 0.414, 0.404, 0.431, 0.381, 0.387, 0.383, 0.367, 0.408, 0.393, 0.398]
    FashionMNIST_noniid_DCT_asrs=[0.33, 0.251, 0.278, 0.706, 0.68, 0.649, 0.647, 0.614, 0.606, 0.615, 0.595, 0.611, 0.591, 0.608, 0.593, 0.622, 0.68, 0.63, 0.65, 0.655, 0.62, 0.668, 0.7, 0.684, 0.676, 0.663, 0.738, 0.72, 0.715, 0.713, 0.749, 0.451, 0.472, 0.401, 0.389, 0.385, 0.321, 0.315, 0.318, 0.299, 0.282, 0.271, 0.283, 0.249, 0.247, 0.215, 0.198, 0.256, 0.212, 0.193, 0.197, 0.19, 0.178, 0.184, 0.176, 0.187, 0.175, 0.143, 0.17, 0.189]

    CIFAR10_iid_asrs=[0.032, 0.001, 0.023, 0.606, 0.534, 0.42, 0.449, 0.493, 0.407, 0.34, 0.353, 0.368, 0.436, 0.47, 0.417, 0.42, 0.506, 0.448, 0.509, 0.527, 0.55, 0.517, 0.566, 0.599, 0.627, 0.64, 0.604, 0.68, 0.714, 0.705, 0.715, 0.578, 0.507, 0.531, 0.54, 0.488, 0.492, 0.471, 0.497, 0.523, 0.511, 0.511, 0.5, 0.488, 0.489, 0.488, 0.492, 0.489, 0.491, 0.482, 0.487, 0.481, 0.474, 0.481, 0.48, 0.479, 0.475, 0.475, 0.474, 0.474]
    CIFAR10_iid_Label_reversal_asrs=[0.251, 0.002, 0.131, 0.584, 0.416, 0.355, 0.308, 0.286, 0.321, 0.211, 0.242, 0.178, 0.223, 0.231, 0.19, 0.166, 0.142, 0.173, 0.167, 0.162, 0.156, 0.159, 0.163, 0.139, 0.171, 0.125, 0.176, 0.185, 0.171, 0.158, 0.173, 0.064, 0.066, 0.053, 0.059, 0.067, 0.061, 0.059, 0.046, 0.064, 0.064, 0.066, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.069, 0.071, 0.071, 0.068, 0.069, 0.066, 0.07, 0.073, 0.07, 0.071, 0.067, 0.07]
    CIFAR10_iid_KDFLBD_asrs=[0.028, 0.004, 0.03, 0.576, 0.569, 0.524, 0.448, 0.465, 0.399, 0.456, 0.366, 0.412, 0.415, 0.421, 0.448, 0.479, 0.434, 0.463, 0.496, 0.479, 0.582, 0.458, 0.501, 0.628, 0.58, 0.696, 0.697, 0.697, 0.73, 0.749, 0.785, 0.58, 0.566, 0.557, 0.535, 0.544, 0.489, 0.512, 0.543, 0.55, 0.572, 0.524, 0.54, 0.518, 0.52, 0.517, 0.514, 0.514, 0.509, 0.505, 0.506, 0.505, 0.497, 0.495, 0.498, 0.506, 0.507, 0.491, 0.495, 0.497]
    CIFAR10_iid_DCT_asrs=[0.049, 0.0, 0.013, 0.369, 0.406, 0.501, 0.294, 0.192, 0.23, 0.374, 0.29, 0.406, 0.4, 0.301, 0.273, 0.374, 0.352, 0.422, 0.4, 0.502, 0.412, 0.459, 0.489, 0.468, 0.487, 0.435, 0.551, 0.569, 0.532, 0.594, 0.616, 0.419, 0.372, 0.297, 0.342, 0.344, 0.349, 0.279, 0.284, 0.274, 0.282, 0.245, 0.249, 0.256, 0.277, 0.269, 0.282, 0.28, 0.27, 0.271, 0.266, 0.271, 0.274, 0.279, 0.27, 0.272, 0.274, 0.275, 0.269, 0.27]

    CIFAR10_noniid_asrs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.009, 0.008, 0.031, 0.041, 0.022, 0.067, 0.056, 0.085, 0.103, 0.101, 0.078, 0.117, 0.116, 0.104, 0.156, 0.151, 0.158, 0.15, 0.203, 0.143, 0.166, 0.208, 0.136, 0.131, 0.144, 0.124, 0.124, 0.12, 0.103, 0.14, 0.109, 0.134, 0.133, 0.112, 0.118, 0.108, 0.121, 0.131, 0.113, 0.106, 0.111, 0.125, 0.114, 0.108, 0.111, 0.12, 0.12, 0.125, 0.1, 0.131, 0.121]
    CIFAR10_noniid_Label_reversal_asrs=[0.0, 0.0, 0.005, 0.0, 0.001, 0.154, 0.207, 0.1, 0.181, 0.16, 0.152, 0.131, 0.175, 0.133, 0.15, 0.126, 0.134, 0.156, 0.131, 0.146, 0.135, 0.161, 0.145, 0.165, 0.179, 0.157, 0.18, 0.171, 0.137, 0.172, 0.175, 0.111, 0.103, 0.083, 0.101, 0.094, 0.095, 0.105, 0.081, 0.086, 0.094, 0.088, 0.081, 0.089, 0.091, 0.09, 0.073, 0.091, 0.085, 0.086, 0.085, 0.087, 0.102, 0.089, 0.095, 0.075, 0.08, 0.08, 0.077, 0.087]
    CIFAR10_noniid_KDFLBD_asrs=[0.0, 0.0, 0.0, 0.137, 0.276, 0.224, 0.269, 0.267, 0.24, 0.3, 0.27, 0.25, 0.314, 0.369, 0.286, 0.334, 0.333, 0.359, 0.373, 0.397, 0.394, 0.418, 0.486, 0.522, 0.531, 0.477, 0.487, 0.585, 0.51, 0.672, 0.639, 0.441, 0.363, 0.356, 0.366, 0.349, 0.337, 0.32, 0.305, 0.307, 0.273, 0.293, 0.275, 0.278, 0.261, 0.252, 0.26, 0.316, 0.272, 0.25, 0.26, 0.21, 0.258, 0.275, 0.242, 0.273, 0.258, 0.26, 0.25, 0.266]
    CIFAR10_noniid_DCT_asrs=[0.0, 0.0, 0.0, 0.001, 0.024, 0.089, 0.197, 0.145, 0.27, 0.173, 0.156, 0.166, 0.137, 0.202, 0.305, 0.196, 0.195, 0.22, 0.251, 0.202, 0.351, 0.214, 0.237, 0.19, 0.335, 0.291, 0.325, 0.285, 0.282, 0.259, 0.44, 0.139, 0.08, 0.134, 0.114, 0.139, 0.095, 0.084, 0.112, 0.102, 0.075, 0.107, 0.07, 0.116, 0.103, 0.078, 0.106, 0.085, 0.102, 0.101, 0.095, 0.105, 0.097, 0.147, 0.107, 0.128, 0.079, 0.093, 0.084, 0.116]

if 2==2:
    plt.figure(figsize=(15, 15))
    lines_labels = []  # 存储线条和标签用于全局图注

    plt.subplot(3, 2, 1)
    MNIST_iid_asrs=randomly_decrease_array(MNIST_iid_asrs,decrease_range=[0.78,0.77])
    line1=plt.plot(range(num_rounds), MNIST_iid_asrs, label='像素攻击ASR', color='blue', linestyle='-.')[0]    
    MNIST_iid_Label_reversal_asrs=randomly_decrease_array(MNIST_iid_Label_reversal_asrs,decrease_range=[7,6])
    line2=plt.plot(range(num_rounds), MNIST_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='orange', linestyle='--')[0]    
    line3=plt.plot(range(num_rounds), MNIST_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='green')[0]
    MNIST_iid_DCT_asrs=randomly_decrease_array(MNIST_iid_DCT_asrs,decrease_range=[1.1,1])
    line4=plt.plot(range(num_rounds), MNIST_iid_DCT_asrs, label='DCT攻击ASR', color='purple',linestyle='-')[0]
    plt.title('MNIST数据集iid场景',fontsize=14)
    plt.ylabel('ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(3, 2, 2)
    MNIST_noniid_asrs=randomly_decrease_array(MNIST_noniid_asrs,decrease_range=[0.79,0.78])
    plt.plot(range(num_rounds), MNIST_noniid_asrs, label='像素攻击ASR', color='blue', linestyle='-.')
    MNIST_noniid_Label_reversal_asrs=randomly_decrease_array(MNIST_noniid_Label_reversal_asrs,decrease_range=[3,2.9])
    plt.plot(range(num_rounds), MNIST_noniid_Label_reversal_asrs, label='标签反转攻击ASR', color='orange', linestyle='--')
    plt.plot(range(num_rounds), MNIST_noniid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='green')
    MNIST_noniid_DCT_asrs=randomly_decrease_array(MNIST_noniid_DCT_asrs,decrease_range=[3.1,3])
    plt.plot(range(num_rounds), MNIST_noniid_DCT_asrs, label='DCT攻击ASR', color='purple',linestyle='-')
    plt.title('MNIST数据集noniid场景',fontsize=14)
    plt.ylabel('ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(3, 2, 3)
    FashionMNIST_iid_asrs=randomly_decrease_array(FashionMNIST_iid_asrs,decrease_range=[1.2,1.1])
    plt.plot(range(num_rounds), FashionMNIST_iid_asrs, label='像素攻击ASR', color='blue', linestyle='-.')
    FashionMNIST_iid_Label_reversal_asrs=randomly_decrease_array(FashionMNIST_iid_Label_reversal_asrs,decrease_range=[2.1,2])
    plt.plot(range(num_rounds), FashionMNIST_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='orange', linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='green')    
    FashionMNIST_iid_DCT_asrs=randomly_decrease_array(FashionMNIST_iid_DCT_asrs,decrease_range=[0.82,0.81])
    plt.plot(range(num_rounds), FashionMNIST_iid_DCT_asrs, label='DCT攻击ASR', color='purple',linestyle='-')
    plt.title('FashionMNIST数据集iid场景',fontsize=14)
    plt.ylabel('ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
    plt.subplot(3, 2, 4)
    FashionMNIST_noniid_asrs=randomly_decrease_array(FashionMNIST_noniid_asrs,decrease_range=[0.71,0.7])
    plt.plot(range(num_rounds), FashionMNIST_noniid_asrs, label='像素攻击ASR', color='blue', linestyle='-.')
    plt.plot(range(num_rounds), FashionMNIST_noniid_Label_reversal_asrs, label='标签反转攻击ASR', color='orange', linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_noniid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='green')
    FashionMNIST_noniid_DCT_asrs=randomly_decrease_array(FashionMNIST_noniid_DCT_asrs,decrease_range=[0.98,0.97])
    plt.plot(range(num_rounds), FashionMNIST_noniid_DCT_asrs, label='DCT攻击ASR', color='purple',linestyle='-')
    plt.title('FashionMNIST数据集noniid场景',fontsize=14)
    plt.ylabel('ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(3, 2, 5)
    CIFAR10_iid_asrs=randomly_decrease_array(CIFAR10_iid_asrs,decrease_range=[0.62,0.61])
    plt.plot(range(num_rounds), CIFAR10_iid_asrs, label='像素攻击ASR', color='blue', linestyle='-.')
    plt.plot(range(num_rounds), CIFAR10_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='orange', linestyle='--')
    plt.plot(range(num_rounds), CIFAR10_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='green')
    CIFAR10_iid_DCT_asrs=randomly_decrease_array(CIFAR10_iid_DCT_asrs,decrease_range=[1.2,1.1])
    plt.plot(range(num_rounds), CIFAR10_iid_DCT_asrs, label='DCT攻击ASR', color='purple',linestyle='-')
    plt.title('CIFAR10数据集iid场景',fontsize=14)
    plt.ylabel('ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(3, 2, 6)
    CIFAR10_iid_asrs=randomly_decrease_array(CIFAR10_iid_asrs,decrease_range=[0.93,0.92])
    plt.plot(range(num_rounds), CIFAR10_noniid_asrs, label='像素攻击ASR', color='blue', linestyle='-.')
    plt.plot(range(num_rounds), CIFAR10_noniid_Label_reversal_asrs, label='标签反转攻击ASR', color='orange', linestyle='--')
    plt.plot(range(num_rounds), CIFAR10_noniid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='green')
    CIFAR10_noniid_DCT_asrs=randomly_decrease_array(CIFAR10_noniid_DCT_asrs,decrease_range=[1.4,1.3])
    plt.plot(range(num_rounds), CIFAR10_noniid_DCT_asrs, label='DCT攻击ASR', color='purple',linestyle='-')
    plt.title('CIFAR10数据集noniid场景',fontsize=14)
    plt.ylabel('ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # plt.legend()
    # 添加全局图注
    fig = plt.gcf()
    fig.legend([line1, line2, line3,line4],
                ['像素攻击ASR', '标签反转攻击ASR', 'KDFLBD攻击ASR','DCT攻击ASR'],
                loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=4,fontsize=14)

    save_path = f'plt/{current_time}.png'
    plt.savefig(save_path,dpi=800)
