import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Continuous-time PINN
    Input: Time t
    Output: Predicted state (x, y, z)
    """
    def __init__(self, hidden_dim=64, num_layers=4):
        super(PINN, self).__init__()
        
        layers = []
        # Input Layer (Input dimension is 1, i.e., time t)
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden Layer
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Output Layer (Output dimension is 3, i.e., x, y, z)
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

def physics_loss(model, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    t.requires_grad_(True)
    xyz = model(t)
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    
    # Differentiate with respect to x, y, and z, respectively. The returned shapes are all (N, 1).
    dx_dt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    dz_dt = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True)[0]
    
    f_x = dx_dt - sigma * (y - x)
    f_y = dy_dt - (x * (rho - z) - y)
    f_z = dz_dt - (x * y - beta * z)
    
    loss_p = torch.mean(f_x**2 + f_y**2 + f_z**2)
    return loss_p

# Quickly Test Code
if __name__ == "__main__":
    model = PINN()
    # Generate a random point in time.
    dummy_t = torch.rand(10, 1) 
    out = model(dummy_t)
    p_loss = physics_loss(model, dummy_t)
    print(f"Output shape: {out.shape}")
    print(f"Initial Physics Loss: {p_loss.item():.4f}")
