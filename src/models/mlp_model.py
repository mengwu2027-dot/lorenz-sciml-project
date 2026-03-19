import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Baseline Multilayer Perceptron (MLP)
    Used to learn the discrete-time step mapping of the Lorenz system: state(t+1) = MLP(state(t))
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, num_layers=3):
        super(MLP, self).__init__()
        
        layers = []
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh()) # Using Tanh ensures the continuity and smoothness of derivatives—a critical aspect of SciML.
        
        # Hidden Layer
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        # Output Layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward Propagation
        Input: x (Current state, shape: [batch_size, 3])
        Output: Predicted next state or state change (shape: [batch_size, 3])
        """
        return self.net(x)

# Quickly Test Code
if __name__ == "__main__":
    # Instantiate the model and print its structure.
    model = MLP()
    print(model)
    
    # Randomly generate a batch of data to test forward propagation.
    dummy_input = torch.randn(10, 3) 
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expected Output: torch.Size([10, 3])
