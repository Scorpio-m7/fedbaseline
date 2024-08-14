import torch    

# 假设你已经定义了模型结构，比如 net = MyModel()
# 加载模型参数
state_dict = torch.load('pth/client_0_round_0_weights.pth')
# 假设你要查看 'conv1.weight' 参数
# print(state_dict)
conv1_weight = state_dict['fc1.weight']
print(conv1_weight)

def visualize_weights(state_dict):
    for param_tensor in state_dict:
        if 'weight' in param_tensor:  # 仅可视化权重参数
            plt.figure(figsize=(10, 5))
            plt.hist(state_dict[param_tensor].cpu().numpy().flatten(), bins=50)
            plt.title(f"Histogram of {param_tensor}")
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.show()

# 加载模型的state_dict并可视化
visualize_weights(state_dict)

