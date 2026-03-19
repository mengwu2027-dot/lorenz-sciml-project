import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Import MLP from the model file.
from models.mlp_model import MLP

def load_data(data_path):
    """Load the generated real data and construct input-output pairs."""
    data_dict = np.load(data_path, allow_pickle=True).item()
    data = data_dict['data']
    
    # Construct the pair (X, Y), where X is the current step and Y is the immediately following step.
    X = data[:-1, :]
    Y = data[1:, :]
    
    # Convert to PyTorch Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    return X_tensor, Y_tensor, data

def main():
    # 1. Set Paths and Hyperparameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'lorenz_ground_truth.npy')
    epochs = 1000
    learning_rate = 1e-3

    # 2. Prepare Data
    print("Loading data...")
    X, Y, raw_data = load_data(data_path)
    
    # 3. Initialize the model, loss function, and optimizer.
    model = MLP(input_dim=3, hidden_dim=64, output_dim=3, num_layers=3)
    criterion = nn.MSELoss() # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward Propagation
        predictions = model(X)
        
        # Calculate Loss
        loss = criterion(predictions, Y)
        
        # Backpropagation and Optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    print("Training complete!")

    # 5. Testing and Visualization (Autoregressive Forecasting)
    # We provide an initial point and let the MLP continuously predict the future trajectory on its own to see if it diverges.
    print("Simulating future trajectory...")
    test_steps = 2000
    current_state = X[0:1] # Take the first time step as the initial state.
    predicted_trajectory = [current_state.detach().numpy().flatten()]

    with torch.no_grad():
        for _ in range(test_steps - 1):
            next_state = model(current_state)
            predicted_trajectory.append(next_state.numpy().flatten())
            current_state = next_state # Use the prediction results as input for the next step (autoregression).

    predicted_trajectory = np.array(predicted_trajectory)

    # Plot and compare the first 2,000 steps.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(raw_data[:test_steps, 0], raw_data[:test_steps, 1], raw_data[:test_steps, 2], label='Ground Truth', color='blue', alpha=0.6, lw=1)
    ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], label='MLP Prediction', color='red', alpha=0.8, lw=1, linestyle='--')
    
    ax.set_title("Lorenz Attractor: Ground Truth vs Baseline MLP")
    ax.legend()
    
    # Ensure that the `figures` folder exists and save high-resolution images.
    figures_dir = os.path.join(current_dir, '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, 'mlp_prediction.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()
