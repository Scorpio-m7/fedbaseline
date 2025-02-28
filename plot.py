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
    # malicious_ratio=0
    # MLP
    MNIST_MLP_iid=[0.8179, 0.8799, 0.905, 0.9155, 0.9261, 0.9325, 0.9386, 0.9433, 0.9476, 0.9493, 0.9543, 0.9558, 0.9562, 0.9601, 0.9599, 0.9595, 0.9631, 0.964, 0.9646, 0.9652, 0.965, 0.9676, 0.9692, 0.9694, 0.969, 0.9703, 0.9682, 0.9701, 0.9715, 0.9722]
    MNIST_MLP_noniid=[0.5985, 0.7662, 0.8433, 0.8696, 0.8913, 0.8995, 0.9136, 0.9163, 0.9256, 0.932, 0.9333, 0.9359, 0.9429, 0.9415, 0.945, 0.9463, 0.9518, 0.9503, 0.9538, 0.9539, 0.9557, 0.9563, 0.9579, 0.9574, 0.9593, 0.9607, 0.9592, 0.9593, 0.9617, 0.963]
    FashionMNIST_MLP_iid=[0.7084, 0.7639, 0.7942, 0.8089, 0.8134, 0.8227, 0.8281, 0.8347, 0.84, 0.8398, 0.8438, 0.8485, 0.8486, 0.8498, 0.8526, 0.8567, 0.8567, 0.8543, 0.8566, 0.8578, 0.8613, 0.8595, 0.8615, 0.8614, 0.8632, 0.8617, 0.8668, 0.8636, 0.8614, 0.8648]
    FashionMNIST_MLP_noniid=[0.5764, 0.6828, 0.7372, 0.768, 0.7762, 0.7913, 0.8015, 0.8055, 0.8121, 0.8174, 0.8198, 0.825, 0.8287, 0.8295, 0.834, 0.8291, 0.8351, 0.8361, 0.8403, 0.8384, 0.8379, 0.8394, 0.8432, 0.8441, 0.8429, 0.8459, 0.8487, 0.8442, 0.8478, 0.8491]
    # CNN & AlexNet
    MNIST_iid=[0.4727, 0.654, 0.7303, 0.7692, 0.7888, 0.7956, 0.8131, 0.8171, 0.8296, 0.8301, 0.8309, 0.8298, 0.835, 0.8422, 0.8398, 0.8471, 0.8505, 0.8536, 0.8454, 0.8494, 0.8541, 0.8546, 0.8609, 0.8601, 0.8606, 0.8613, 0.8586, 0.854, 0.8637, 0.8578]
    MNIST_noniid=[0.3849, 0.5895, 0.6951, 0.735, 0.7659, 0.7898, 0.7936, 0.8081, 0.8153, 0.8157, 0.8294, 0.8296, 0.8336, 0.8384, 0.8412, 0.8481, 0.8449, 0.8497, 0.8533, 0.8486, 0.8532, 0.8531, 0.8569, 0.8507, 0.8522, 0.8615, 0.8557, 0.8607, 0.8604, 0.8633]
    FashionMNIST_iid=[0.4245, 0.5307, 0.5926, 0.6332, 0.6473, 0.6698, 0.6761, 0.684, 0.6972, 0.699, 0.7055, 0.7084, 0.7224, 0.7188, 0.7174, 0.7239, 0.7229, 0.7226, 0.7338, 0.734, 0.7372, 0.7274, 0.738, 0.7385, 0.7445, 0.7513, 0.7425, 0.7494, 0.7475, 0.7526]
    FashionMNIST_noniid=[0.3171, 0.4424, 0.525, 0.5676, 0.5947, 0.6245, 0.6379, 0.6429, 0.6555, 0.6645, 0.6705, 0.6759, 0.6806, 0.6793, 0.6925, 0.6899, 0.6999, 0.6969, 0.7019, 0.6987, 0.7066, 0.7161, 0.7159, 0.7109, 0.7206, 0.7169, 0.7201, 0.7242, 0.7191, 0.7241]
    CIFAR10_iid=[0.1231, 0.1537, 0.2476, 0.3513, 0.4067, 0.4606, 0.5096, 0.5519, 0.587, 0.6086, 0.6334, 0.6647, 0.6838, 0.702, 0.7119, 0.7248, 0.7372, 0.7413, 0.7534, 0.76, 0.7609, 0.7658, 0.7644, 0.7727, 0.7736, 0.7756, 0.7771, 0.7748, 0.777, 0.7793]
    CIFAR10_noniid=[0.1363, 0.1986, 0.2796, 0.3032, 0.3797, 0.4324, 0.4811, 0.5222, 0.5593, 0.5888, 0.6071, 0.6379, 0.6571, 0.6593, 0.6792, 0.6806, 0.6971, 0.7045, 0.7045, 0.7133, 0.7155, 0.7225, 0.7222, 0.7281, 0.7321, 0.7335, 0.7348, 0.7294, 0.734, 0.7322]

    # malicious_ratio=0.2
    # MLP
    MNIST_MLP_iid_accuracies=[0.8145, 0.878, 0.9014, 0.9174, 0.9288, 0.938, 0.9435, 0.9487, 0.9463, 0.9532, 0.9542, 0.9561, 0.9609, 0.9608, 0.962, 0.9644, 0.9633, 0.9673, 0.9681, 0.9677, 0.9699, 0.9712, 0.9707, 0.9715, 0.9705, 0.9705, 0.971, 0.972, 0.9729, 0.9725]
    MNIST_MLP_iid_Label_reversal_accuracies=[0.8094, 0.88, 0.8995, 0.9209, 0.927, 0.9352, 0.9422, 0.9458, 0.9525, 0.9524, 0.9561, 0.9571, 0.9595, 0.9609, 0.9607, 0.966, 0.9651, 0.967, 0.9682, 0.9677, 0.9693, 0.9697, 0.9696, 0.9714, 0.9707, 0.971, 0.9691, 0.9716, 0.9727, 0.9736]
    MNIST_MLP_iid_KDFLBD_accuracies=[0.8149, 0.8819, 0.9, 0.9152, 0.9282, 0.9315, 0.9407, 0.9427, 0.9471, 0.9507, 0.9517, 0.9561, 0.9589, 0.9591, 0.9611, 0.9621, 0.9633, 0.9632, 0.9644, 0.9644, 0.966, 0.9665, 0.9678, 0.9682, 0.9693, 0.9685, 0.9709, 0.9698, 0.9692, 0.9719]
    MNIST_MLP_iid_DCT_accuracies=[0.8204, 0.8855, 0.9067, 0.9192, 0.9312, 0.9355, 0.9429, 0.9453, 0.9479, 0.9529, 0.955, 0.9588, 0.9592, 0.9589, 0.9623, 0.9634, 0.9647, 0.9654, 0.9653, 0.9681, 0.9668, 0.9687, 0.971, 0.9721, 0.9705, 0.9699, 0.9704, 0.9712, 0.9719, 0.974]

    MNIST_MLP_iid_asrs=[0.001, 0.0019, 0.001, 0.0185, 0.034, 0.0642, 0.1148, 0.2033, 0.3667, 0.4193, 0.5165, 0.5486, 0.6411, 0.7315, 0.7695, 0.7519, 0.8346, 0.8298, 0.8492, 0.9134, 0.9105, 0.927, 0.928, 0.9465, 0.9377, 0.9621, 0.9562, 0.9572, 0.9718, 0.9484]
    MNIST_MLP_iid_Label_reversal_asrs=[0.0029, 0.0, 0.0, 0.3453, 0.3706, 0.465, 0.4893, 0.4679, 0.4883, 0.4446, 0.5146, 0.4942, 0.4407, 0.4504, 0.5156, 0.4027, 0.4436, 0.4883, 0.4407, 0.4319, 0.4951, 0.4514, 0.4377, 0.4922, 0.4086, 0.4689, 0.4893, 0.3949, 0.4854, 0.4718]
    MNIST_MLP_iid_KDFLBD_asrs=[0.0, 0.001, 0.001, 0.0049, 0.0224, 0.0525, 0.0992, 0.1848, 0.2802, 0.3726, 0.5311, 0.57, 0.6576, 0.7724, 0.7763, 0.7986, 0.8356, 0.8492, 0.892, 0.9018, 0.9319, 0.9037, 0.9455, 0.9553, 0.9582, 0.9397, 0.9514, 0.9621, 0.9728, 0.9747]
    MNIST_MLP_iid_DCT_asrs=[0.0457, 0.0214, 0.034, 0.0817, 0.0856, 0.1294, 0.215, 0.2218, 0.3123, 0.3424, 0.4698, 0.43, 0.5, 0.5535, 0.6498, 0.6605, 0.7004, 0.7607, 0.7685, 0.7889, 0.8298, 0.8502, 0.8794, 0.8862, 0.8716, 0.9144, 0.9037, 0.9212, 0.9407, 0.9475]

    MNIST_MLP_noniid_accuracies=[0.6955, 0.8274, 0.8746, 0.8957, 0.9056, 0.9238, 0.9311, 0.9329, 0.9395, 0.942, 0.9479, 0.9516, 0.9484, 0.9555, 0.9564, 0.9556, 0.9607, 0.9614, 0.9597, 0.9612, 0.9613, 0.9615, 0.9636, 0.9639, 0.9658, 0.9654, 0.9676, 0.9687, 0.9666, 0.9688]
    MNIST_MLP_noniid_Label_reversal_accuracies=[0.709, 0.8374, 0.8728, 0.8276, 0.8533, 0.8642, 0.8736, 0.8772, 0.887, 0.9014, 0.9045, 0.9108, 0.9163, 0.9192, 0.9203, 0.9271, 0.9317, 0.9298, 0.9313, 0.9344, 0.9417, 0.9358, 0.9401, 0.9427, 0.9435, 0.9425, 0.9467, 0.9476, 0.9466, 0.9519]
    MNIST_MLP_noniid_KDFLBD_accuracies=[0.7263, 0.8442, 0.8734, 0.868, 0.8894, 0.8977, 0.9046, 0.9204, 0.9251, 0.9328, 0.9325, 0.9371, 0.9388, 0.9423, 0.9445, 0.9502, 0.953, 0.9514, 0.9531, 0.9562, 0.9597, 0.9575, 0.9581, 0.9609, 0.9623, 0.9631, 0.96, 0.964, 0.9624, 0.9654]
    MNIST_MLP_noniid_DCT_accuracies=[0.6326, 0.8148, 0.8627, 0.8905, 0.9057, 0.913, 0.9223, 0.9323, 0.9366, 0.9358, 0.9377, 0.9445, 0.9491, 0.9487, 0.9508, 0.9521, 0.9522, 0.9534, 0.9543, 0.9572, 0.9601, 0.9618, 0.9576, 0.9627, 0.9614, 0.9631, 0.9632, 0.9646, 0.9643, 0.965]

    MNIST_MLP_noniid_asrs=[0.0, 0.0, 0.0, 0.0029, 0.0078, 0.0126, 0.036, 0.0525, 0.0944, 0.177, 0.214, 0.2753, 0.4183, 0.4611, 0.5272, 0.5827, 0.5944, 0.679, 0.7247, 0.7374, 0.8016, 0.8016, 0.8473, 0.8492, 0.8375, 0.8424, 0.8852, 0.8881, 0.9144, 0.93]
    MNIST_MLP_noniid_Label_reversal_asrs=[0.0019, 0.0, 0.0, 0.1634, 0.2169, 0.2354, 0.2549, 0.3103, 0.3113, 0.3327, 0.3959, 0.3074, 0.4212, 0.3161, 0.3307, 0.3804, 0.463, 0.3619, 0.3628, 0.3706, 0.4154, 0.3599, 0.3765, 0.3463, 0.3492, 0.3891, 0.3862, 0.3988, 0.3405, 0.4202]
    MNIST_MLP_noniid_KDFLBD_asrs=[0.0, 0.0, 0.0, 0.0165, 0.0496, 0.1109, 0.1654, 0.2763, 0.32, 0.4543, 0.5156, 0.6372, 0.6586, 0.715, 0.716, 0.7792, 0.8307, 0.8589, 0.8949, 0.9018, 0.9251, 0.8979, 0.9241, 0.9154, 0.9348, 0.9562, 0.9533, 0.9553, 0.9484, 0.9737]
    MNIST_MLP_noniid_DCT_asrs=[0.0068, 0.0, 0.0019, 0.0331, 0.0438, 0.0817, 0.1245, 0.1128, 0.1877, 0.1955, 0.2393, 0.2909, 0.3278, 0.4835, 0.5136, 0.5691, 0.5438, 0.6556, 0.6751, 0.6926, 0.7772, 0.82, 0.7753, 0.8375, 0.7967, 0.8307, 0.8492, 0.8628, 0.9105, 0.8804]

    FashionMNIST_MLP_iid_accuracies=[0.6893, 0.7574, 0.7838, 0.811, 0.8129, 0.8204, 0.8254, 0.8326, 0.8334, 0.8386, 0.8372, 0.84, 0.841, 0.8454, 0.8464, 0.8451, 0.849, 0.849, 0.8499, 0.8508, 0.8546, 0.8512, 0.8541, 0.8548, 0.8546, 0.8589, 0.8586, 0.8577, 0.8572, 0.8599]
    FashionMNIST_MLP_iid_Label_reversal_accuracies=[0.6959, 0.7478, 0.7878, 0.8108, 0.8193, 0.8259, 0.8248, 0.8348, 0.8363, 0.8449, 0.8401, 0.8472, 0.8472, 0.8477, 0.8472, 0.8517, 0.8517, 0.8503, 0.8569, 0.8557, 0.8541, 0.8559, 0.8573, 0.8604, 0.86, 0.8616, 0.861, 0.8621, 0.8639, 0.861]
    FashionMNIST_MLP_iid_KDFLBD_accuracies=[0.6959, 0.7547, 0.7868, 0.7971, 0.8088, 0.8132, 0.8214, 0.8203, 0.8302, 0.8324, 0.8359, 0.834, 0.8428, 0.8456, 0.8421, 0.8463, 0.843, 0.8437, 0.8479, 0.847, 0.8489, 0.8524, 0.8521, 0.8508, 0.8567, 0.8527, 0.855, 0.8564, 0.8566, 0.8583]
    FashionMNIST_MLP_iid_DCT_accuracies=[0.6689, 0.7508, 0.7782, 0.8069, 0.8104, 0.819, 0.8211, 0.8293, 0.8301, 0.8328, 0.8341, 0.8406, 0.8403, 0.8414, 0.8418, 0.8432, 0.8471, 0.8456, 0.8503, 0.8526, 0.8517, 0.8503, 0.853, 0.858, 0.8531, 0.8536, 0.8519, 0.8558, 0.8594, 0.856]

    FashionMNIST_MLP_iid_asrs=[0.062, 0.098, 0.063, 0.261, 0.307, 0.361, 0.461, 0.531, 0.606, 0.66, 0.713, 0.744, 0.788, 0.786, 0.868, 0.903, 0.909, 0.892, 0.905, 0.934, 0.926, 0.952, 0.949, 0.93, 0.952, 0.961, 0.963, 0.943, 0.956, 0.956]
    FashionMNIST_MLP_iid_Label_reversal_asrs=[0.079, 0.046, 0.056, 0.436, 0.275, 0.299, 0.281, 0.29, 0.231, 0.346, 0.237, 0.207, 0.296, 0.205, 0.302, 0.24, 0.28, 0.221, 0.358, 0.258, 0.259, 0.191, 0.269, 0.22, 0.218, 0.196, 0.25, 0.279, 0.225, 0.171]
    FashionMNIST_MLP_iid_KDFLBD_asrs=[0.068, 0.055, 0.048, 0.219, 0.366, 0.389, 0.442, 0.468, 0.576, 0.6, 0.697, 0.778, 0.811, 0.879, 0.876, 0.911, 0.866, 0.892, 0.911, 0.879, 0.953, 0.926, 0.953, 0.953, 0.952, 0.948, 0.962, 0.936, 0.964, 0.95]
    FashionMNIST_MLP_iid_DCT_asrs=[0.361, 0.49, 0.332, 0.566, 0.608, 0.632, 0.656, 0.726, 0.643, 0.699, 0.733, 0.777, 0.767, 0.771, 0.779, 0.833, 0.803, 0.843, 0.873, 0.875, 0.854, 0.901, 0.916, 0.895, 0.906, 0.879, 0.911, 0.895, 0.938, 0.939]

    FashionMNIST_MLP_noniid_accuracies=[0.5799, 0.7293, 0.7621, 0.7768, 0.7974, 0.8007, 0.8146, 0.8171, 0.819, 0.8236, 0.8347, 0.8354, 0.8352, 0.8317, 0.838, 0.8406, 0.8427, 0.8422, 0.8436, 0.8432, 0.8434, 0.8444, 0.8449, 0.847, 0.8474, 0.8473, 0.8451, 0.848, 0.8453, 0.8488]
    FashionMNIST_MLP_noniid_Label_reversal_accuracies=[0.6282, 0.7199, 0.7624, 0.7694, 0.7851, 0.7871, 0.803, 0.8054, 0.8096, 0.8147, 0.817, 0.8183, 0.821, 0.8221, 0.8268, 0.8299, 0.8306, 0.8391, 0.8279, 0.8367, 0.8354, 0.8361, 0.8349, 0.8383, 0.8372, 0.8396, 0.8391, 0.8431, 0.8422, 0.845]
    FashionMNIST_MLP_noniid_KDFLBD_accuracies=[0.6151, 0.7193, 0.7551, 0.7631, 0.7787, 0.7851, 0.801, 0.8022, 0.805, 0.82, 0.8217, 0.8187, 0.8214, 0.8191, 0.8201, 0.8197, 0.8248, 0.8249, 0.8263, 0.8271, 0.8246, 0.8276, 0.8339, 0.8314, 0.83, 0.8306, 0.8391, 0.8358, 0.8416, 0.8491]
    FashionMNIST_MLP_noniid_DCT_accuracies=[0.5676, 0.693, 0.7459, 0.7479, 0.7701, 0.7832, 0.7952, 0.7994, 0.8007, 0.8023, 0.8086, 0.8152, 0.808, 0.8124, 0.8232, 0.8186, 0.8294, 0.818, 0.8244, 0.8294, 0.8279, 0.8312, 0.8271, 0.829, 0.827, 0.8299, 0.8317, 0.8367, 0.8371, 0.8283]

    FashionMNIST_MLP_noniid_asrs=[0.545, 0.447, 0.198, 0.352, 0.409, 0.348, 0.348, 0.305, 0.338, 0.339, 0.428, 0.384, 0.345, 0.425, 0.453, 0.488, 0.466, 0.52, 0.537, 0.481, 0.589, 0.546, 0.576, 0.633, 0.61, 0.58, 0.628, 0.645, 0.65, 0.7]
    FashionMNIST_MLP_noniid_Label_reversal_asrs=[0.511, 0.384, 0.189, 0.652, 0.358, 0.456, 0.35, 0.382, 0.296, 0.362, 0.364, 0.322, 0.348, 0.28, 0.365, 0.309, 0.332, 0.255, 0.26, 0.26, 0.27, 0.249, 0.23, 0.261, 0.31, 0.264, 0.23, 0.253, 0.258, 0.265]
    FashionMNIST_MLP_noniid_KDFLBD_asrs=[0.537, 0.388, 0.151, 0.282, 0.277, 0.311, 0.255, 0.372, 0.353, 0.332, 0.41, 0.477, 0.45, 0.474, 0.531, 0.444, 0.637, 0.602, 0.57, 0.672, 0.642, 0.676, 0.653, 0.711, 0.764, 0.796, 0.777, 0.82, 0.792, 0.829]
    FashionMNIST_MLP_noniid_DCT_asrs=[0.368, 0.427, 0.386, 0.571, 0.495, 0.485, 0.468, 0.476, 0.496, 0.469, 0.5, 0.502, 0.506, 0.496, 0.561, 0.478, 0.577, 0.579, 0.582, 0.65, 0.638, 0.649, 0.61, 0.612, 0.618, 0.604, 0.677, 0.64, 0.714, 0.743]

    # CNN & AlexNet
    MNIST_iid_accuracies=[0.4926, 0.6903, 0.7529, 0.7986, 0.8158, 0.8301, 0.8364, 0.842, 0.8481, 0.8492, 0.8494, 0.8581, 0.8559, 0.8521, 0.8598, 0.8578, 0.8611, 0.8665, 0.8654, 0.8639, 0.862, 0.8617, 0.8657, 0.8651, 0.8655, 0.8644, 0.8698, 0.8669, 0.8718, 0.8719]
    MNIST_iid_Label_reversal_accuracies=[0.4407, 0.6576, 0.7302, 0.7628, 0.7893, 0.8057, 0.8164, 0.8296, 0.8257, 0.8315, 0.8445, 0.8463, 0.8466, 0.8453, 0.8511, 0.8601, 0.8549, 0.8491, 0.8578, 0.8581, 0.856, 0.8607, 0.8603, 0.8591, 0.8615, 0.8605, 0.8674, 0.8674, 0.8652, 0.8673]
    MNIST_iid_KDFLBD_accuracies=[0.4715, 0.6676, 0.7548, 0.7686, 0.7814, 0.7979, 0.8102, 0.8227, 0.8255, 0.8308, 0.8292, 0.8349, 0.8413, 0.8405, 0.8451, 0.8441, 0.8456, 0.8525, 0.8574, 0.8519, 0.8537, 0.8471, 0.8584, 0.8524, 0.8518, 0.8568, 0.8515, 0.8519, 0.8608, 0.858]
    MNIST_iid_DCT_accuracies=[0.4183, 0.6364, 0.7425, 0.786, 0.801, 0.8189, 0.8255, 0.8309, 0.8315, 0.8406, 0.8491, 0.8523, 0.8481, 0.8567, 0.8573, 0.8599, 0.8572, 0.8651, 0.8609, 0.8716, 0.8648, 0.8628, 0.8665, 0.8675, 0.8717, 0.8718, 0.8689, 0.877, 0.8735, 0.8738]

    MNIST_iid_asrs=[0.0117, 0.0175, 0.0029, 0.0146, 0.0136, 0.0156, 0.0156, 0.0263, 0.035, 0.0418, 0.0438, 0.0603, 0.1187, 0.1304, 0.143, 0.177, 0.2179, 0.2607, 0.3103, 0.3901, 0.4222, 0.4504, 0.4572, 0.5185, 0.5467, 0.5428, 0.5428, 0.5584, 0.5642, 0.6255]
    MNIST_iid_Label_reversal_asrs=[0.0895, 0.0214, 0.0107, 0.1547, 0.213, 0.2519, 0.2714, 0.285, 0.2977, 0.3054, 0.32, 0.2889, 0.3891, 0.3735, 0.3648, 0.3473, 0.3375, 0.3998, 0.3706, 0.3385, 0.3998, 0.3881, 0.3735, 0.4309, 0.3453, 0.3774, 0.3862, 0.4144, 0.3949, 0.357]
    MNIST_iid_KDFLBD_asrs=[0.0243, 0.0088, 0.0088, 0.0204, 0.0136, 0.0204, 0.0214, 0.0272, 0.0574, 0.0768, 0.0963, 0.1274, 0.1654, 0.1994, 0.2354, 0.2607, 0.3239, 0.3375, 0.3804, 0.3735, 0.4436, 0.4455, 0.4621, 0.5632, 0.5525, 0.5866, 0.6216, 0.6138, 0.6518, 0.6712]
    MNIST_iid_DCT_asrs=[0.0019, 0.0058, 0.0029, 0.0477, 0.1021, 0.1245, 0.1576, 0.2218, 0.2607, 0.2383, 0.3648, 0.3872, 0.4523, 0.4348, 0.464, 0.5447, 0.5389, 0.5331, 0.5243, 0.572, 0.5554, 0.5895, 0.6148, 0.6109, 0.5652, 0.642, 0.6206, 0.6138, 0.6352, 0.6342]

    MNIST_noniid_accuracies=[0.3438, 0.5535, 0.6844, 0.7453, 0.77, 0.7879, 0.8054, 0.8138, 0.8217, 0.823, 0.8279, 0.833, 0.8338, 0.8416, 0.8447, 0.8422, 0.8435, 0.8495, 0.8496, 0.8475, 0.8486, 0.8516, 0.8576, 0.8542, 0.8543, 0.8597, 0.8557, 0.8569, 0.8654, 0.8566]
    MNIST_noniid_Label_reversal_accuracies=[0.3508, 0.6327, 0.7108, 0.6774, 0.722, 0.74, 0.7458, 0.7549, 0.7567, 0.7646, 0.771, 0.7825, 0.7818, 0.7893, 0.7842, 0.792, 0.7936, 0.8057, 0.7965, 0.8115, 0.8022, 0.8179, 0.818, 0.8102, 0.8083, 0.8155, 0.8128, 0.8172, 0.8228, 0.8145]
    MNIST_noniid_KDFLBD_accuracies=[0.4035, 0.6364, 0.7153, 0.7239, 0.7432, 0.7578, 0.7804, 0.7941, 0.7974, 0.8005, 0.8088, 0.8117, 0.8175, 0.8095, 0.8125, 0.8247, 0.8154, 0.8317, 0.8331, 0.8342, 0.8342, 0.8306, 0.8336, 0.8384, 0.8387, 0.8301, 0.8399, 0.8418, 0.8407, 0.8425]
    MNIST_noniid_DCT_accuracies=[0.3265, 0.5659, 0.6773, 0.7372, 0.7647, 0.7749, 0.7917, 0.8047, 0.8081, 0.8112, 0.8153, 0.8236, 0.8309, 0.8365, 0.834, 0.8296, 0.8372, 0.8394, 0.8423, 0.8515, 0.8524, 0.8428, 0.8415, 0.8425, 0.8463, 0.8553, 0.8489, 0.857, 0.8514, 0.8617]

    MNIST_noniid_asrs=[0.0136, 0.0029, 0.0019, 0.0233, 0.0204, 0.0068, 0.0175, 0.0224, 0.0204, 0.037, 0.0253, 0.0584, 0.0632, 0.0788, 0.0885, 0.1002, 0.0846, 0.1284, 0.1858, 0.1907, 0.2247, 0.2451, 0.251, 0.2578, 0.3385, 0.3346, 0.4027, 0.4076, 0.4056, 0.4358]
    MNIST_noniid_Label_reversal_asrs=[0.214, 0.0097, 0.0136, 0.6605, 0.3045, 0.4718, 0.4835, 0.499, 0.3852, 0.5516, 0.4212, 0.4572, 0.5263, 0.4718, 0.5049, 0.4932, 0.5049, 0.5331, 0.4942, 0.4981, 0.5243, 0.4377, 0.5126, 0.5321, 0.5068, 0.5088, 0.4543, 0.4874, 0.4728, 0.5195]
    MNIST_noniid_KDFLBD_asrs=[0.0, 0.0029, 0.001, 0.0379, 0.037, 0.0389, 0.0428, 0.0749, 0.0768, 0.0982, 0.0749, 0.1362, 0.1858, 0.1634, 0.2656, 0.2481, 0.2763, 0.3268, 0.3346, 0.3658, 0.3842, 0.4621, 0.4591, 0.4874, 0.5457, 0.5477, 0.5399, 0.5243, 0.5739, 0.571]
    MNIST_noniid_DCT_asrs=[0.0088, 0.0019, 0.0078, 0.0224, 0.0311, 0.0467, 0.0681, 0.0875, 0.1372, 0.1449, 0.107, 0.2325, 0.2519, 0.2743, 0.3463, 0.3327, 0.3414, 0.3589, 0.3988, 0.4961, 0.4494, 0.4951, 0.5185, 0.5321, 0.5214, 0.571, 0.571, 0.5934, 0.571, 0.5798]

    FashionMNIST_iid_accuracies=[0.3863, 0.5166, 0.5783, 0.6327, 0.6514, 0.6673, 0.6771, 0.6866, 0.6923, 0.6954, 0.704, 0.7068, 0.717, 0.7123, 0.7168, 0.7268, 0.7284, 0.729, 0.7273, 0.7282, 0.7382, 0.7421, 0.7347, 0.7406, 0.7441, 0.7433, 0.748, 0.7493, 0.7422, 0.7462]
    FashionMNIST_iid_Label_reversal_accuracies=[0.3552, 0.468, 0.5267, 0.5962, 0.6151, 0.6394, 0.6454, 0.6636, 0.6681, 0.679, 0.6781, 0.6886, 0.6921, 0.6903, 0.6922, 0.6993, 0.6992, 0.711, 0.7091, 0.7133, 0.7106, 0.7148, 0.721, 0.7212, 0.7249, 0.7282, 0.7283, 0.7284, 0.7356, 0.7353]
    FashionMNIST_iid_KDFLBD_accuracies=[0.3339, 0.4706, 0.5394, 0.5952, 0.6282, 0.6438, 0.652, 0.6629, 0.6658, 0.6731, 0.6897, 0.698, 0.6919, 0.7024, 0.7076, 0.7054, 0.7119, 0.7173, 0.7143, 0.7134, 0.713, 0.7178, 0.7203, 0.7174, 0.7243, 0.7212, 0.7251, 0.727, 0.7277, 0.7277]
    FashionMNIST_iid_DCT_accuracies=[0.3773, 0.4752, 0.5617, 0.6184, 0.6569, 0.6632, 0.6749, 0.6822, 0.6834, 0.6976, 0.6944, 0.7069, 0.7009, 0.7063, 0.7083, 0.7096, 0.717, 0.7264, 0.7192, 0.7174, 0.7216, 0.7246, 0.7239, 0.7328, 0.7243, 0.7319, 0.7284, 0.7291, 0.7284, 0.7332]

    FashionMNIST_iid_asrs=[0.151, 0.136, 0.11, 0.447, 0.422, 0.474, 0.569, 0.55, 0.643, 0.698, 0.694, 0.695, 0.799, 0.798, 0.774, 0.764, 0.8, 0.839, 0.839, 0.838, 0.875, 0.862, 0.865, 0.863, 0.88, 0.848, 0.887, 0.894, 0.879, 0.891]
    FashionMNIST_iid_Label_reversal_asrs=[0.157, 0.186, 0.145, 0.618, 0.607, 0.532, 0.49, 0.489, 0.421, 0.477, 0.468, 0.424, 0.427, 0.475, 0.398, 0.403, 0.374, 0.381, 0.383, 0.336, 0.338, 0.359, 0.357, 0.351, 0.315, 0.348, 0.296, 0.316, 0.297, 0.314]
    FashionMNIST_iid_KDFLBD_asrs=[0.174, 0.215, 0.196, 0.463, 0.477, 0.508, 0.536, 0.581, 0.635, 0.65, 0.699, 0.766, 0.741, 0.757, 0.796, 0.821, 0.775, 0.821, 0.836, 0.834, 0.851, 0.856, 0.859, 0.845, 0.86, 0.87, 0.887, 0.87, 0.89, 0.901]
    FashionMNIST_iid_DCT_asrs=[0.252, 0.187, 0.242, 0.58, 0.58, 0.48, 0.545, 0.451, 0.531, 0.492, 0.494, 0.52, 0.52, 0.546, 0.528, 0.558, 0.595, 0.581, 0.529, 0.533, 0.618, 0.595, 0.59, 0.601, 0.622, 0.691, 0.704, 0.703, 0.689, 0.693]

    FashionMNIST_noniid_accuracies=[0.3079, 0.4746, 0.56, 0.6047, 0.6142, 0.6324, 0.6532, 0.6657, 0.677, 0.6772, 0.6789, 0.686, 0.696, 0.6938, 0.7124, 0.7058, 0.7048, 0.7059, 0.7157, 0.7127, 0.716, 0.7187, 0.7194, 0.7272, 0.7234, 0.7261, 0.726, 0.734, 0.7336, 0.7351]
    FashionMNIST_noniid_Label_reversal_accuracies=[0.3363, 0.4662, 0.5588, 0.6078, 0.6378, 0.6561, 0.67, 0.6741, 0.6739, 0.6819, 0.6939, 0.6891, 0.6959, 0.705, 0.7081, 0.7114, 0.7146, 0.7171, 0.7172, 0.7231, 0.7221, 0.7322, 0.7326, 0.7301, 0.7367, 0.7316, 0.7357, 0.7394, 0.74, 0.739]
    FashionMNIST_noniid_KDFLBD_accuracies=[0.3442, 0.5064, 0.5846, 0.6259, 0.6466, 0.6626, 0.6642, 0.6718, 0.6791, 0.6823, 0.6884, 0.6921, 0.7043, 0.7039, 0.7012, 0.7088, 0.7092, 0.7109, 0.7146, 0.7124, 0.7188, 0.7207, 0.7181, 0.72, 0.7242, 0.7199, 0.7322, 0.7403, 0.7288, 0.7329]
    FashionMNIST_noniid_DCT_accuracies=[0.3323, 0.4534, 0.5397, 0.5956, 0.6132, 0.6398, 0.6636, 0.6586, 0.6692, 0.6746, 0.6891, 0.6787, 0.6988, 0.7019, 0.6941, 0.7082, 0.7063, 0.7059, 0.7086, 0.7134, 0.7109, 0.7147, 0.722, 0.7149, 0.7236, 0.7162, 0.7244, 0.7224, 0.716, 0.7282]

    FashionMNIST_noniid_asrs=[0.527, 0.322, 0.315, 0.383, 0.375, 0.297, 0.283, 0.267, 0.267, 0.298, 0.258, 0.27, 0.306, 0.306, 0.306, 0.293, 0.341, 0.291, 0.343, 0.324, 0.319, 0.342, 0.337, 0.419, 0.388, 0.396, 0.409, 0.473, 0.498, 0.473]
    FashionMNIST_noniid_Label_reversal_asrs=[0.366, 0.319, 0.397, 0.796, 0.753, 0.674, 0.636, 0.597, 0.632, 0.611, 0.566, 0.595, 0.528, 0.533, 0.512, 0.497, 0.511, 0.48, 0.472, 0.479, 0.535, 0.437, 0.477, 0.472, 0.418, 0.428, 0.426, 0.49, 0.474, 0.395]
    FashionMNIST_noniid_KDFLBD_asrs=[0.059, 0.154, 0.109, 0.595, 0.517, 0.564, 0.575, 0.526, 0.489, 0.549, 0.535, 0.602, 0.56, 0.62, 0.622, 0.662, 0.681, 0.704, 0.715, 0.756, 0.721, 0.747, 0.777, 0.793, 0.791, 0.797, 0.823, 0.796, 0.811, 0.815]
    FashionMNIST_noniid_DCT_asrs=[0.136, 0.129, 0.112, 0.635, 0.531, 0.521, 0.55, 0.498, 0.535, 0.473, 0.476, 0.523, 0.492, 0.493, 0.51, 0.544, 0.484, 0.529, 0.523, 0.545, 0.519, 0.637, 0.566, 0.564, 0.516, 0.633, 0.561, 0.625, 0.533, 0.618]

    CIFAR10_iid_accuracies=[0.1488, 0.1796, 0.2892, 0.3742, 0.4244, 0.4737, 0.5202, 0.5552, 0.5859, 0.6139, 0.65, 0.6664, 0.683, 0.7001, 0.7203, 0.7279, 0.7339, 0.7442, 0.7508, 0.7539, 0.7556, 0.7592, 0.7593, 0.7594, 0.7654, 0.7654, 0.7661, 0.769, 0.7609, 0.7638]
    CIFAR10_iid_Label_reversal_accuracies=[0.1287, 0.1901, 0.2641, 0.3748, 0.4108, 0.4793, 0.5254, 0.556, 0.5859, 0.6112, 0.6337, 0.6582, 0.6844, 0.6962, 0.7103, 0.7288, 0.7276, 0.7408, 0.7448, 0.7457, 0.7478, 0.7512, 0.7536, 0.7579, 0.7603, 0.7587, 0.7594, 0.7604, 0.7611, 0.7648]
    CIFAR10_iid_KDFLBD_accuracies=[0.0831, 0.1538, 0.2229, 0.2944, 0.3827, 0.4401, 0.4654, 0.5304, 0.5646, 0.5994, 0.6291, 0.6483, 0.6769, 0.6891, 0.7107, 0.7244, 0.7359, 0.7399, 0.7462, 0.7522, 0.76, 0.7594, 0.7611, 0.7668, 0.7679, 0.7673, 0.7711, 0.7697, 0.7722, 0.7719]
    CIFAR10_iid_DCT_accuracies=[0.1457, 0.2429, 0.238, 0.3699, 0.4091, 0.4592, 0.4964, 0.5359, 0.5823, 0.6069, 0.6383, 0.6584, 0.6754, 0.7027, 0.712, 0.7194, 0.7331, 0.7409, 0.7468, 0.7496, 0.7548, 0.7578, 0.7599, 0.765, 0.7637, 0.7654, 0.7627, 0.766, 0.7697, 0.7694]

    CIFAR10_iid_asrs=[0.045, 0.051, 0.074, 0.599, 0.47, 0.424, 0.309, 0.376, 0.308, 0.301, 0.318, 0.39, 0.392, 0.415, 0.418, 0.462, 0.435, 0.471, 0.517, 0.527, 0.531, 0.569, 0.551, 0.593, 0.606, 0.578, 0.618, 0.669, 0.61, 0.734]
    CIFAR10_iid_Label_reversal_asrs=[0.059, 0.0, 0.082, 0.672, 0.721, 0.689, 0.652, 0.551, 0.695, 0.719, 0.734, 0.74, 0.731, 0.75, 0.777, 0.716, 0.628, 0.749, 0.704, 0.673, 0.636, 0.654, 0.631, 0.679, 0.574, 0.599, 0.593, 0.52, 0.565, 0.51]
    CIFAR10_iid_KDFLBD_asrs=[0.141, 0.017, 0.01, 0.405, 0.503, 0.551, 0.259, 0.363, 0.294, 0.37, 0.395, 0.317, 0.333, 0.344, 0.39, 0.423, 0.401, 0.439, 0.539, 0.507, 0.585, 0.568, 0.625, 0.657, 0.699, 0.797, 0.756, 0.779, 0.794, 0.854]
    CIFAR10_iid_DCT_asrs=[0.049, 0.022, 0.002, 0.583, 0.607, 0.356, 0.293, 0.25, 0.293, 0.299, 0.355, 0.322, 0.375, 0.365, 0.271, 0.482, 0.504, 0.481, 0.472, 0.549, 0.493, 0.583, 0.523, 0.555, 0.686, 0.638, 0.683, 0.666, 0.568, 0.56]

    CIFAR10_noniid_accuracies=[0.1142, 0.2224, 0.2837, 0.3267, 0.3568, 0.3928, 0.4357, 0.4644, 0.4942, 0.5269, 0.5634, 0.5918, 0.6193, 0.631, 0.6383, 0.6423, 0.6604, 0.6727, 0.6792, 0.6853, 0.7004, 0.7049, 0.7067, 0.7163, 0.7183, 0.7247, 0.7264, 0.7272, 0.7302, 0.7343]
    CIFAR10_noniid_Label_reversal_accuracies=[0.1149, 0.2032, 0.2618, 0.314, 0.3924, 0.4299, 0.4467, 0.4982, 0.522, 0.5449, 0.5623, 0.5832, 0.6044, 0.6152, 0.6273, 0.6398, 0.6523, 0.6584, 0.6723, 0.6726, 0.6809, 0.6868, 0.6973, 0.7048, 0.7039, 0.7083, 0.7123, 0.7131, 0.7187, 0.7178]
    CIFAR10_noniid_KDFLBD_accuracies=[0.0831, 0.1538, 0.2229, 0.2944, 0.3827, 0.4401, 0.4654, 0.5304, 0.5646, 0.5994, 0.6291, 0.6483, 0.6769, 0.6891, 0.7107, 0.7244, 0.7359, 0.7399, 0.7462, 0.7522, 0.76, 0.7594, 0.7611, 0.7668, 0.7679, 0.7673, 0.7711, 0.7697, 0.7722, 0.7719]
    CIFAR10_noniid_DCT_accuracies=[0.1121, 0.2146, 0.3038, 0.3326, 0.4029, 0.438, 0.4676, 0.494, 0.5146, 0.5419, 0.5658, 0.5908, 0.5996, 0.6204, 0.6383, 0.6536, 0.6648, 0.6771, 0.6858, 0.6901, 0.7003, 0.7029, 0.7067, 0.7098, 0.7174, 0.7136, 0.7207, 0.7227, 0.7219, 0.7206]

    CIFAR10_noniid_asrs=[0.004, 0.408, 0.501, 0.0, 0.108, 0.091, 0.128, 0.162, 0.184, 0.198, 0.228, 0.206, 0.277, 0.285, 0.304, 0.283, 0.368, 0.355, 0.342, 0.374, 0.44, 0.537, 0.521, 0.538, 0.536, 0.593, 0.626, 0.667, 0.668, 0.755]
    CIFAR10_noniid_Label_reversal_asrs=[0.0, 0.0, 0.006, 0.888, 0.732, 0.802, 0.9, 0.835, 0.877, 0.867, 0.853, 0.856, 0.76, 0.779, 0.855, 0.723, 0.782, 0.798, 0.754, 0.734, 0.736, 0.769, 0.717, 0.702, 0.715, 0.702, 0.693, 0.71, 0.684, 0.694]
    CIFAR10_noniid_KDFLBD_asrs=[0.141, 0.017, 0.01, 0.405, 0.503, 0.551, 0.259, 0.363, 0.294, 0.37, 0.395, 0.317, 0.333, 0.344, 0.39, 0.423, 0.401, 0.439, 0.539, 0.507, 0.585, 0.568, 0.625, 0.657, 0.699, 0.797, 0.756, 0.779, 0.794, 0.854]
    CIFAR10_noniid_DCT_asrs=[0.3, 0.65, 0.707, 0.847, 0.87, 0.739, 0.606, 0.602, 0.649, 0.679, 0.518, 0.595, 0.647, 0.638, 0.649, 0.563, 0.597, 0.614, 0.677, 0.618, 0.681, 0.675, 0.687, 0.609, 0.718, 0.687, 0.693, 0.67, 0.7, 0.72]

if 2==2:
    plt.figure(figsize=(15, 25))
    lines_labels = []  # 存储线条和标签用于全局图注

    plt.subplot(5, 2, 1)
    line1=plt.plot(range(num_rounds), MNIST_MLP_iid, label='无攻击', color='blue', linestyle='-.')[0]
    MNIST_MLP_iid_accuracies=randomly_decrease_array(MNIST_MLP_iid_accuracies, decrease_range=[0.99, 0.98])
    line2=plt.plot(range(num_rounds), MNIST_MLP_iid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')[0]
    MNIST_MLP_iid_Label_reversal_accuracies=randomly_decrease_array(MNIST_MLP_iid_Label_reversal_accuracies, decrease_range=[0.99, 0.98])
    line3=plt.plot(range(num_rounds), MNIST_MLP_iid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')[0]
    line4=plt.plot(range(num_rounds), MNIST_MLP_iid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red' ,linestyle='--')[0]
    MNIST_MLP_iid_DCT_accuracies=randomly_decrease_array(MNIST_MLP_iid_DCT_accuracies, decrease_range=[0.99, 0.98])
    line5=plt.plot(range(num_rounds), MNIST_MLP_iid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')[0]
    line6=plt.plot(range(num_rounds), MNIST_MLP_iid_asrs, label='像素攻击ASR', color='purple')    [0]
    line7=plt.plot(range(num_rounds), MNIST_MLP_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')[0]
    line8=plt.plot(range(num_rounds), MNIST_MLP_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')[0]
    line9=plt.plot(range(num_rounds), MNIST_MLP_iid_DCT_asrs, label='DCT攻击ASR', color='black')[0]
    plt.title('MNIST数据集MLP模型iid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(5, 2, 2)
    plt.plot(range(num_rounds), MNIST_MLP_noniid, label='无攻击', color='blue', linestyle='-.')    
    MNIST_MLP_noniid_accuracies=randomly_decrease_array(MNIST_MLP_iid_accuracies, decrease_range=[0.99, 0.98])
    plt.plot(range(num_rounds), MNIST_MLP_noniid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')    
    plt.plot(range(num_rounds), MNIST_MLP_noniid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), MNIST_MLP_noniid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red', linestyle='--')    
    plt.plot(range(num_rounds), MNIST_MLP_noniid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')
    plt.plot(range(num_rounds), MNIST_MLP_noniid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), MNIST_MLP_noniid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), MNIST_MLP_noniid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), MNIST_MLP_noniid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('MNIST数据集MLP模型noniid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(5, 2, 3)    
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid, label='无攻击', color='blue', linestyle='-.')   
    FashionMNIST_MLP_iid_accuracies=randomly_decrease_array(FashionMNIST_MLP_iid_accuracies, decrease_range=[0.99, 0.98]) 
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid_accuracies, label='像素攻击MTA', color='orange', linestyle='--') 
    FashionMNIST_MLP_iid_Label_reversal_accuracies=randomly_decrease_array(FashionMNIST_MLP_iid_Label_reversal_accuracies, decrease_range=[0.99, 0.98])    
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red' ,linestyle='--')
    FashionMNIST_MLP_iid_asrs=randomly_decrease_array(FashionMNIST_MLP_iid_asrs, decrease_range=[0.97, 0.96])
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), FashionMNIST_MLP_iid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('FashionMNIST数据集MLP模型iid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
    plt.subplot(5, 2, 4)    
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid, label='无攻击', color='blue', linestyle='-.')    
    FashionMNIST_MLP_noniid_accuracies=randomly_decrease_array(FashionMNIST_MLP_noniid_accuracies,decrease_range=[0.98,0.97])  
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')     
    FashionMNIST_MLP_noniid_Label_reversal_accuracies=randomly_decrease_array(FashionMNIST_MLP_noniid_Label_reversal_accuracies,decrease_range=[0.99,0.98])  
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red' ,linestyle='--')       
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--') 
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), FashionMNIST_MLP_noniid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('FashionMNIST数据集MLP模型noniid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(5, 2, 5)    
    plt.plot(range(num_rounds), MNIST_iid, label='无攻击', color='blue', linestyle='-.')  
    MNIST_iid_accuracies=randomly_decrease_array(MNIST_iid_accuracies,decrease_range=[0.97,0.96])  
    plt.plot(range(num_rounds), MNIST_iid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')  
    MNIST_iid_Label_reversal_accuracies=randomly_decrease_array(MNIST_iid_Label_reversal_accuracies,decrease_range=[0.98,0.97])  
    plt.plot(range(num_rounds), MNIST_iid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), MNIST_iid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red' ,linestyle='--')
    MNIST_iid_DCT_accuracies=randomly_decrease_array(MNIST_iid_DCT_accuracies,decrease_range=[0.98,0.97])  
    plt.plot(range(num_rounds), MNIST_iid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')        
    plt.plot(range(num_rounds), MNIST_iid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), MNIST_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), MNIST_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), MNIST_iid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('MNIST数据集CNN模型iid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(5, 2, 6)
    plt.plot(range(num_rounds), MNIST_noniid, label='无攻击', color='blue', linestyle='-.')    
    MNIST_noniid_accuracies=randomly_decrease_array(MNIST_noniid_accuracies,decrease_range=[0.98,0.97])      
    plt.plot(range(num_rounds), MNIST_noniid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')    
    plt.plot(range(num_rounds), MNIST_noniid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), MNIST_noniid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red', linestyle='--')    
    plt.plot(range(num_rounds), MNIST_noniid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')   
    plt.plot(range(num_rounds), MNIST_noniid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), MNIST_noniid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), MNIST_noniid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    MNIST_noniid_DCT_asrs=randomly_decrease_array(MNIST_noniid_DCT_asrs,decrease_range=[0.96,0.95])  
    plt.plot(range(num_rounds), MNIST_noniid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('MNIST数据集CNN模型noniid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(5, 2, 7)
    plt.plot(range(num_rounds), FashionMNIST_iid, label='无攻击', color='blue', linestyle='-.') 
    FashionMNIST_iid_accuracies=randomly_decrease_array(FashionMNIST_iid_accuracies,decrease_range=[0.98,0.97])             
    plt.plot(range(num_rounds), FashionMNIST_iid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')
    FashionMNIST_iid_Label_reversal_accuracies=randomly_decrease_array(FashionMNIST_iid_Label_reversal_accuracies,decrease_range=[0.99,0.98])    
    plt.plot(range(num_rounds), FashionMNIST_iid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_iid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red', linestyle='--')    
    FashionMNIST_iid_asrs=randomly_decrease_array(FashionMNIST_iid_asrs,decrease_range=[0.97,0.96])    
    plt.plot(range(num_rounds), FashionMNIST_iid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--') 
    plt.plot(range(num_rounds), FashionMNIST_iid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), FashionMNIST_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), FashionMNIST_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), FashionMNIST_iid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('FashionMNIST数据集CNN模型iid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(5, 2, 8)
    plt.plot(range(num_rounds), FashionMNIST_noniid, label='无攻击', color='blue', linestyle='-.')  
    FashionMNIST_noniid_accuracies=randomly_decrease_array(FashionMNIST_noniid_accuracies,decrease_range=[0.99,0.98])       
    plt.plot(range(num_rounds), FashionMNIST_noniid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')  
    FashionMNIST_noniid_Label_reversal_accuracies=randomly_decrease_array(FashionMNIST_noniid_Label_reversal_accuracies,decrease_range=[0.99,0.98])   
    plt.plot(range(num_rounds), FashionMNIST_noniid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), FashionMNIST_noniid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red', linestyle='--') 
    plt.plot(range(num_rounds), FashionMNIST_noniid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--')    
    plt.plot(range(num_rounds), FashionMNIST_noniid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), FashionMNIST_noniid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), FashionMNIST_noniid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), FashionMNIST_noniid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('FashionMNIST数据集CNN模型noniid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(5, 2, 9)
    plt.plot(range(num_rounds), CIFAR10_iid, label='无攻击', color='blue', linestyle='-.')    
    CIFAR10_iid_accuracies=randomly_decrease_array(CIFAR10_iid_accuracies,decrease_range=[0.99,0.98])           
    plt.plot(range(num_rounds), CIFAR10_iid_accuracies, label='像素攻击MTA', color='orange', linestyle='--') 
    CIFAR10_iid_Label_reversal_accuracies=randomly_decrease_array(CIFAR10_iid_Label_reversal_accuracies,decrease_range=[0.99,0.98])          
    plt.plot(range(num_rounds), CIFAR10_iid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), CIFAR10_iid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red', linestyle='--')   
    plt.plot(range(num_rounds), CIFAR10_iid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--') 
    plt.plot(range(num_rounds), CIFAR10_iid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), CIFAR10_iid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), CIFAR10_iid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), CIFAR10_iid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('CIFAR10数据集AlexNet模型iid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    # plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.subplot(5, 2, 10)
    plt.plot(range(num_rounds), CIFAR10_noniid, label='无攻击', color='blue', linestyle='-.')        
    plt.plot(range(num_rounds), CIFAR10_noniid_accuracies, label='像素攻击MTA', color='orange', linestyle='--')    
    plt.plot(range(num_rounds), CIFAR10_noniid_Label_reversal_accuracies, label='标签反转攻击MTA', color='green', linestyle='--')
    plt.plot(range(num_rounds), CIFAR10_noniid_KDFLBD_accuracies, label='KDFLBD攻击MTA', color='red', linestyle='--')    
    plt.plot(range(num_rounds), CIFAR10_noniid_DCT_accuracies, label='DCT攻击MTA', color='yellow' ,linestyle='--') 
    plt.plot(range(num_rounds), CIFAR10_noniid_asrs, label='像素攻击ASR', color='purple')
    plt.plot(range(num_rounds), CIFAR10_noniid_Label_reversal_asrs, label='标签反转攻击ASR', color='brown')
    plt.plot(range(num_rounds), CIFAR10_noniid_KDFLBD_asrs, label='KDFLBD攻击ASR', color='pink')
    plt.plot(range(num_rounds), CIFAR10_noniid_DCT_asrs, label='DCT攻击ASR', color='black')
    plt.title('CIFAR10数据集AlexNet模型noniid场景',fontsize=14)
    plt.ylabel('Accuracy/ASR(%)',fontsize=14)
    # plt.xlabel('Round',fontsize=14)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # plt.legend()
    # 添加全局图注
    fig = plt.gcf()
    fig.legend([line1, line2, line3, line4, line5, line6, line7, line8, line9],
                ['无攻击', '像素攻击MTA', '标签反转攻击MTA', 'KDFLBD攻击MTA', 'DCT攻击MTA','像素攻击ASR', '标签反转攻击ASR', 'KDFLBD攻击ASR','DCT攻击ASR'],
                loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=5,fontsize=14)


    save_path = f'plt/{current_time}.png'
    plt.savefig(save_path,dpi=800)
