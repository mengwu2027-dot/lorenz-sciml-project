import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 从你刚刚写的模型文件中引入 MLP
from models.mlp_model import MLP

def load_data(data_path):
    """加载生成好的真实数据，并构造输入-输出对"""
    data_dict = np.load(data_path, allow_pickle=True).item()
    data = data_dict['data']
    
    # 构造 (X, Y) 对：X 是当前步，Y 是紧接着的下一步
    X = data[:-1, :]
    Y = data[1:, :]
    
    # 转换为 PyTorch 的 Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    return X_tensor, Y_tensor, data

def main():
    # 1. 设置路径与超参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'lorenz_ground_truth.npy')
    epochs = 1000
    learning_rate = 1e-3

    # 2. 准备数据
    print("Loading data...")
    X, Y, raw_data = load_data(data_path)
    
    # 3. 初始化模型、损失函数和优化器
    model = MLP(input_dim=3, hidden_dim=64, output_dim=3, num_layers=3)
    criterion = nn.MSELoss() # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. 训练循环
    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(X)
        
        # 计算 Loss
        loss = criterion(predictions, Y)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    print("Training complete!")

    # 5. 测试与可视化 (自回归预测)
    # 我们给定一个初始点，让 MLP 自己连续预测未来的轨迹，看看它会不会发散
    print("Simulating future trajectory...")
    test_steps = 2000
    current_state = X[0:1] # 取第一个时间步作为初始状态
    predicted_trajectory = [current_state.detach().numpy().flatten()]

    with torch.no_grad():
        for _ in range(test_steps - 1):
            next_state = model(current_state)
            predicted_trajectory.append(next_state.numpy().flatten())
            current_state = next_state # 将预测结果作为下一步的输入 (自回归)

    predicted_trajectory = np.array(predicted_trajectory)

    # 画图对比前 2000 步
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(raw_data[:test_steps, 0], raw_data[:test_steps, 1], raw_data[:test_steps, 2], label='Ground Truth', color='blue', alpha=0.6, lw=1)
    ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], label='MLP Prediction', color='red', alpha=0.8, lw=1, linestyle='--')
    
   ax.set_title("Lorenz Attractor: Ground Truth vs Baseline MLP")
    ax.legend()
    
    # 确保 figures 文件夹存在并保存高清晰度图片
    figures_dir = os.path.join(current_dir, '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, 'mlp_prediction.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()
