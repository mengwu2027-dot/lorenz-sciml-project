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
   
    
    train_steps = 200
    t_data = torch.tensor(data_dict['t'][:train_steps], dtype=torch.float32).view(-1, 1)
    xyz_data = torch.tensor(data_dict['data'][:train_steps, :], dtype=torch.float32)
    
    # 【新增关键步骤】：对数据进行标准化计算
    xyz_mean = xyz_data.mean(dim=0)
    xyz_std = xyz_data.std(dim=0)
    # 网络要拟合的目标变成了标准化后的数据 (均值为0，方差为1)
    xyz_data_normalized = (xyz_data - xyz_mean) / xyz_std
    
    # 初始化模型
    model = PINN(hidden_dim=64, num_layers=6) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    epochs = 10000 
    lambda_physics = 1e-3 # 加上标准化后，1e-3 就不会引发爆炸了

    print("Starting PINN training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # (1) Data Loss: 让模型去拟合【标准化后】的坐标点
        xyz_pred_normalized = model(t_data)
        loss_data = mse_loss(xyz_pred_normalized, xyz_data_normalized)
        
        # (2) Physics Loss: 核心修改
        t_physics = t_data.clone().requires_grad_(True)
        xyz_pred_norm_phys = model(t_physics)
        
        # 必须将网络预测出的标准化数据【反向还原】成真实的物理量，才能代入物理方程！
        x_real = xyz_pred_norm_phys[:, 0:1] * xyz_std[0] + xyz_mean[0]
        y_real = xyz_pred_norm_phys[:, 1:2] * xyz_std[1] + xyz_mean[1]
        z_real = xyz_pred_norm_phys[:, 2:3] * xyz_std[2] + xyz_mean[2]
        
        # 利用自动微分求导 (对真实的物理量求导)
        dx_dt = torch.autograd.grad(x_real, t_physics, grad_outputs=torch.ones_like(x_real), create_graph=True)[0]
        dy_dt = torch.autograd.grad(y_real, t_physics, grad_outputs=torch.ones_like(y_real), create_graph=True)[0]
        dz_dt = torch.autograd.grad(z_real, t_physics, grad_outputs=torch.ones_like(z_real), create_graph=True)[0]
        
        # 代入 Lorenz 方程
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        f_x = dx_dt - sigma * (y_real - x_real)
        f_y = dy_dt - (x_real * (rho - z_real) - y_real)
        f_z = dz_dt - (x_real * y_real - beta * z_real)
        
        loss_phys = torch.mean(f_x**2 + f_y**2 + f_z**2)
        
        # 组合 Loss
        loss = loss_data + lambda_physics * loss_phys
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {loss.item():.4f} "
                  f"(Data: {loss_data.item():.4f}, Phys: {loss_phys.item():.4f})")
            
    print("Training complete!")

   # 3. 泛化测试与可视化 (加入外推预测)
    print("Generating extrapolation prediction plot...")
    model.eval()
    with torch.no_grad():
        # 【修改这里】：让测试步长达到 400 步 (包含 200步已知 + 200步未知)
        test_steps = 400 
        t_test = torch.tensor(data_dict['t'][:test_steps], dtype=torch.float32).view(-1, 1)
        
        # 网络的预测及反归一化
        predicted_norm = model(t_test)
        predicted_xyz = (predicted_norm * xyz_std + xyz_mean).numpy()
        
        true_xyz = data_dict['data'][:test_steps, :]
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画出前 400 步真实的蓝线
    ax.plot(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2], 
            label='Ground Truth (400 steps)', color='blue', alpha=0.4, lw=1)
            
    # 区分训练区和预测区
    # 1. 训练过的区域 (前 200 步)，用绿色虚线
    ax.plot(predicted_xyz[:200, 0], predicted_xyz[:200, 1], predicted_xyz[:200, 2], 
            label='PINN (Training Region)', color='green', lw=2, linestyle='--')
            
    # 2. 未知的未来区域 (200步 到 400步)，用红色实线展示它的外推能力
    ax.plot(predicted_xyz[200:, 0], predicted_xyz[200:, 1], predicted_xyz[200:, 2], 
            label='PINN (Future Prediction)', color='red', lw=2)
    
    # 画一个点，标记“现在”时刻（也就是数据用完的分界点）
    ax.scatter(*predicted_xyz[199, :], color='black', s=50, label='Prediction Start', zorder=5)

    ax.set_title("Lorenz Attractor: Ground Truth vs PINN")
    ax.legend()
    
    # 保存图片
    figures_dir = os.path.join(current_dir, '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, 'pinn_prediction.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()
    
