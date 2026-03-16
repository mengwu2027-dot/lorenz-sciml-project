import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 引入我们刚才写好的 PINN 模型和物理损失函数
from models.pinn import PINN, physics_loss

def main():
    # 1. 路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'lorenz_ground_truth.npy')
    
    # 加载真实数据
    data_dict = np.load(data_path, allow_pickle=True).item()
    
    # 【细节】：PINN 拟合长时间跨度的混沌系统非常困难。
    # 学术界的标准做法是先在一个较短的时间窗口内展示 PINN 的极高拟合精度。
    # 我们这里取前 500 个时间步（即 5 秒）的轨迹作为训练域。
    train_steps = 500
    t_data = torch.tensor(data_dict['t'][:train_steps], dtype=torch.float32).view(-1, 1)
    xyz_data = torch.tensor(data_dict['data'][:train_steps, :], dtype=torch.float32)
    
    # 2. 初始化模型、优化器与超参数
    model = PINN(hidden_dim=64, num_layers=6) # 网络比 MLP 稍深一点，增强物理拟合能力
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    epochs = 2000 # PINN 需要更多的 epoch 来平衡数据与物理之间的博弈
    lambda_physics = 1e-3 # 物理 Loss 的权重系数

    print("Starting PINN training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # (1) Data Loss: 让模型预测值贴合真实的坐标点
        xyz_pred = model(t_data)
        loss_data = mse_loss(xyz_pred, xyz_data)
        
        # (2) Physics Loss: 将时间 t 输入物理损失函数，计算方程残差
        # 必须 clone 并开启 requires_grad 以供自动微分计算导数
        t_physics = t_data.clone().requires_grad_(True)
        loss_phys = physics_loss(model, t_physics)
        
        # 核心：将数据驱动与物理定律结合
        loss = loss_data + lambda_physics * loss_phys
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {loss.item():.4f} "
                  f"(Data: {loss_data.item():.4f}, Phys: {loss_phys.item():.4f})")
            
    print("Training complete!")

    # 3. 泛化测试与可视化
    print("Generating prediction plot...")
    model.eval()
    with torch.no_grad():
        # 我们故意让模型预测 600 步（超出了它训练时见过的 500 步范围），
        # 看看加上物理约束后，它在未知领域的轨迹会不会像 MLP 那样直接崩溃。
        test_steps = 600
        t_test = torch.tensor(data_dict['t'][:test_steps], dtype=torch.float32).view(-1, 1)
        predicted_xyz = model(t_test).numpy()
        true_xyz = data_dict['data'][:test_steps, :]
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制对比图
    ax.plot(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2], label='Ground Truth', color='blue', alpha=0.6, lw=1)
    ax.plot(predicted_xyz[:, 0], predicted_xyz[:, 1], predicted_xyz[:, 2], label='PINN Prediction', color='green', alpha=0.8, lw=1.5, linestyle='--')
    
    ax.set_title("Lorenz Attractor: Ground Truth vs PINN")
    ax.legend()
    
    # 自动保存到 figures 文件夹
    figures_dir = os.path.join(current_dir, '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, 'pinn_prediction.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()
    
