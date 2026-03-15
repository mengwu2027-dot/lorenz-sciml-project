import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    连续时间物理约束神经网络 (Continuous-time PINN)
    输入: 时间 t
    输出: 预测的状态 (x, y, z)
    """
    def __init__(self, hidden_dim=64, num_layers=4):
        super(PINN, self).__init__()
        
        layers = []
        # 输入层 (输入维度为 1，即时间 t)
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.Tanh())
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # 输出层 (输出维度为 3，即 x, y, z)
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

def physics_loss(model, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    计算物理约束损失 (Physics-Informed Loss)
    利用自动微分 (Autograd) 将 Lorenz 控制方程的残差作为 Loss 加入训练
    """
    # 告诉 PyTorch 我们需要对时间 t 求导
    t.requires_grad_(True)
    
    # 模型预测当前的 (x, y, z)
    xyz = model(t)
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    
    # 核心魔法：使用自动微分求预测值对时间 t 的偏导数 (dx/dt, dy/dt, dz/dt)
    dxyz_dt = torch.autograd.grad(
        xyz, t, 
        grad_outputs=torch.ones_like(xyz),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    dx_dt, dy_dt, dz_dt = dxyz_dt[:, 0:1], dxyz_dt[:, 1:2], dxyz_dt[:, 2:3]
    
    # 根据 Lorenz 微分方程计算残差 (Residual)
    # 完美的物理模型，以下三项应该全部等于 0
    f_x = dx_dt - sigma * (y - x)
    f_y = dy_dt - (x * (rho - z) - y)
    f_z = dz_dt - (x * y - beta * z)
    
    # 物理 Loss 就是方程残差的均方误差
    loss_p = torch.mean(f_x**2 + f_y**2 + f_z**2)
    return loss_p

# 快速测试代码
if __name__ == "__main__":
    model = PINN()
    # 随机生成一个时间点
    dummy_t = torch.rand(10, 1) 
    out = model(dummy_t)
    p_loss = physics_loss(model, dummy_t)
    print(f"Output shape: {out.shape}")
    print(f"Initial Physics Loss: {p_loss.item():.4f}")
