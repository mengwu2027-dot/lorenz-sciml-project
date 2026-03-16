import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 核心：引入常微分方程的求解器（使用 adjoint 伴随方法可以大幅节省显存）
from torchdiffeq import odeint_adjoint as odeint
from models.neural_ode import LorenzODEFunc

def get_batch(t, true_y, batch_time=20, batch_size=32):
    """
    【细节】：在混沌系统中，直接对整个长序列进行积分会导致梯度爆炸。
    标准的训练策略是：每次随机截取一小段短时间序列（比如未来 20 步）来进行训练。
    """
    s = torch.from_numpy(np.random.choice(np.arange(len(t) - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # 取出每个截断的初始状态 (batch_size, 3)
    batch_t = t[:batch_time]  # (batch_time)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # 真实的目标轨迹
    return batch_y0, batch_t, batch_y

def main():
    # 1. 路径与数据加载
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'lorenz_ground_truth.npy')
    
    data_dict = np.load(data_path, allow_pickle=True).item()
    t_data = torch.tensor(data_dict['t'], dtype=torch.float32)
    y_data = torch.tensor(data_dict['data'], dtype=torch.float32)
    
    # 截取前 1000 步用于训练
    train_size = 1000
    t_train = t_data[:train_size]
    y_train = y_data[:train_size]

    # 2. 初始化模型与优化器
    func = LorenzODEFunc(hidden_dim=64)
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 400
    batch_time = 20 # 每次预测未来 20 步

    print("Starting Neural ODE training (this might take a few minutes)...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        batch_y0, batch_t, batch_y = get_batch(t_train, y_train, batch_time=batch_time)
        
        # 魔法发生的地方：将初始状态和时间交给求解器，让它通过神经网络顺藤摸瓜积分出整条轨迹
        pred_y = odeint(func, batch_y0, batch_t)
        
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    print(f"Training complete! Time elapsed: {time.time() - start_time:.2f}s")

    # 3. 泛化测试与可视化 (长序列硬核测试)
    print("Simulating long-term trajectory with trained Neural ODE...")
    func.eval()
    with torch.no_grad():
        # 我们只给模型最初始的一个点，让它自己顺着时间硬生生积分出长达 1500 步的轨迹！
        # 这对于 MLP 是绝对不可能完成的任务。
        test_steps = 1500
        t_test = t_data[:test_steps]
        y0_test = y_data[0] # 取绝对初始点
        
        # 预测整条轨迹
        pred_trajectory = odeint(func, y0_test, t_test).numpy()
        true_trajectory = y_data[:test_steps].numpy()

    # 4. 绘图对比
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], 
            label='Ground Truth', color='blue', alpha=0.6, lw=1)
    # Neural ODE 用醒目的亮橙色
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
            label='Neural ODE Prediction', color='orange', alpha=0.9, lw=1.5, linestyle='-')
    
    ax.set_title("Lorenz Attractor: Ground Truth vs Neural ODE")
    ax.legend()
    
    figures_dir = os.path.join(current_dir, '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, 'neural_ode_prediction.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()
