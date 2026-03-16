import torch
import torch.nn as nn

class LorenzODEFunc(nn.Module):
    """
    Neural ODE 的核心：用神经网络参数化连续时间的导数
    即学习 dh/dt = f_theta(t, h)
    """
    def __init__(self, hidden_dim=64):
        super(LorenzODEFunc, self).__init__()
        
        # 这是一个多层感知机，用来逼近 Lorenz 系统的右侧真实物理方程
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(), # 依然使用平滑的 Tanh
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), # 稍微加深网络结构以捕捉混沌复杂性
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
        
        # 初始化权重：在 Neural ODE 中，如果初始导数太大，
        # 会导致积分器步长变得极小，训练极其缓慢。因此用较小的方差初始化。
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        """
        torchdiffeq 库的硬性要求：forward 必须同时接收 t 和 y。
        t: 当前时间 (标量)
        y: 当前状态 [batch_size, 3]
        返回: 预测的状态变化率 [batch_size, 3]
        """
        # 因为 Lorenz 是自治系统（方程右侧不显式包含时间 t），
        # 所以我们实际上只将状态 y 输入给神经网络。
        return self.net(y)

# 快速测试代码
if __name__ == "__main__":
    func = LorenzODEFunc()
    dummy_t = torch.tensor(0.0)
    dummy_y = torch.randn(10, 3) # 模拟 10 个批次的初始状态
    dy_dt = func(dummy_t, dummy_y)
    
    print(f"Input state shape: {dummy_y.shape}")
    print(f"Derivative (dy/dt) output shape: {dy_dt.shape}")
