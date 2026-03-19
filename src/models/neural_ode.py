import torch
import torch.nn as nn

class LorenzODEFunc(nn.Module):
    """
    The Core of Neural ODEs: Parameterizing Continuous-Time Derivatives Using Neural Networks
    — That is, Learning dh/dt = f_theta(t, h)
    """
    def __init__(self, hidden_dim=64):
        super(LorenzODEFunc, self).__init__()
        
        # This is a multilayer perceptron used to approximate the true physical equations on the right-hand side of the Lorenz system.
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(), # Still using the smooth Tanh.
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), # Slightly deepen the network structure to capture chaotic complexity.
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
        
        # Weight Initialization: In Neural ODEs, if the initial derivative is too large,
        # the integrator step size would become extremely small, resulting in extremely slow training. Therefore, a smaller variance is used for initialization.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        """
        A strict requirement of the `torchdiffeq` library is that the `forward` method must accept both `t` and `y`.
        t: Current time (scalar)
        y: Current state [batch_size, 3]
        Returns: Predicted rate of change of the state [batch_size, 3]
        """
        # Since the Lorenz system is an autonomous system (the right-hand side of the equations does not explicitly contain time *t*),
        # we effectively feed only the state *y* into the neural network.
        return self.net(y)

# Quickly Test Code
if __name__ == "__main__":
    func = LorenzODEFunc()
    dummy_t = torch.tensor(0.0)
    dummy_y = torch.randn(10, 3) # Simulate the initial states of 10 batches.
    dy_dt = func(dummy_t, dummy_y)
    
    print(f"Input state shape: {dummy_y.shape}")
    print(f"Derivative (dy/dt) output shape: {dy_dt.shape}")
