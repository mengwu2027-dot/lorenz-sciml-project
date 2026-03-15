import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    基础的多层感知机 (Baseline MLP)
    用于学习 Lorenz 系统的离散时间步长映射: state(t+1) = MLP(state(t))
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, num_layers=3):
        super(MLP, self).__init__()
        
        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh()) # 使用 Tanh 保证导数的连续和平滑，这在 SciML 中很关键
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播
        输入: x (当前状态, shape: [batch_size, 3])
        输出: 预测的下一个状态或状态变化量 (shape: [batch_size, 3])
        """
        return self.net(x)

# 快速测试代码
if __name__ == "__main__":
    # 实例化模型并打印结构
    model = MLP()
    print(model)
    
    # 随机生成一个批次的数据测试前向传播
    dummy_input = torch.randn(10, 3) 
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # 预期输出: torch.Size([10, 3])
