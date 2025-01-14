import torch
import numpy as np

def calculate_parameter_distance(model1, model2):
    params_1 = model1.parameters()
    params_2 = model2.parameters()
    distance = 0.0

    # 计算模型参数之间的欧氏距离
    for p1, p2 in zip(params_1, params_2):
        distance += torch.norm(p1 - p2)

    return distance.item()


def Krum(client_model_list):
    # 计算每对模型之间的参数差异（欧氏距离）
    num_clients = len(client_model_list)
    distances = np.zeros((num_clients, num_clients))

    for i in range(num_clients):
        for j in range(num_clients):
            model_i = client_model_list[i]
            model_j = client_model_list[j]
            distance = calculate_parameter_distance_from_dicts(model_i, model_j)
            distances[i][j] = distance

    # 使用Krum算法选择最可信的模型索引
    k = num_clients - 1  # Krum算法选择次数
    trusted_indices = Krum_algorithm(distances, k)
    print(trusted_indices)
    return [client_model_list[idx] for idx in trusted_indices]

def Krum_algorithm(distances, k):
    num_clients = distances.shape[0]
    score = np.zeros(num_clients)
    # 计算每个模型的得分
    for i in range(num_clients):
        dist_i = np.delete(distances[i], i)  # 删除自身与自身的距离
        dist_i_sorted = np.sort(dist_i)[:k]  # 选择最小的k-1个距离
        score[i] = np.sum(dist_i_sorted)
    # 选择得分最低的模型
    trusted_indices = np.argsort(score)[:k]
    return trusted_indices


def calculate_parameter_distance_from_dicts(dict1, dict2):
    distance = 0.0
    for key in dict1.keys():
        p1 = dict1[key]
        p2 = dict2[key]
        distance += torch.norm(p1 - p2)
    return distance.item()
