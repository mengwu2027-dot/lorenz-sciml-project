import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Core: Introduction of an Ordinary Differential Equation (ODE) solver (utilizing the adjoint method can significantly reduce GPU memory usage).
from torchdiffeq import odeint_adjoint as odeint
from models.neural_ode import LorenzODEFunc

def get_batch(t, true_y, batch_time=50, batch_size=32):
    """
    In chaotic systems, directly integrating the entire long sequence leads to gradient explosion.
    The standard training strategy is to randomly sample a short time series segment (e.g., the next 20 steps) for training during each iteration.
    """
    s = torch.from_numpy(np.random.choice(np.arange(len(t) - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # Extract the initial state of each truncated segment. (batch_size, 3)
    batch_t = t[:batch_time]  # (batch_time)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # Actual Target Trajectory
    return batch_y0, batch_t, batch_y

def main():
    # 1. Paths and Data Loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'lorenz_ground_truth.npy')
    
    data_dict = np.load(data_path, allow_pickle=True).item()
    t_data = torch.tensor(data_dict['t'], dtype=torch.float32)
    y_data = torch.tensor(data_dict['data'], dtype=torch.float32)
    
    # Extract the first 1,000 steps for training.
    train_size = 1000
    t_train = t_data[:train_size]
    y_train = y_data[:train_size]

    # 2. Initialize the model and optimizer.
    func = LorenzODEFunc(hidden_dim=64)
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 2000
    batch_time = 50 # Predict the next 20 steps each time.

    print("Starting Neural ODE training (this might take a few minutes)...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        batch_y0, batch_t, batch_y = get_batch(t_train, y_train, batch_time=batch_time)
        
        # Handing the initial state and time over to the solver, we allow it—guided by a neural network—to trace out and integrate the entire trajectory.
        pred_y = odeint(func, batch_y0, batch_t)
        
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

    print(f"Training complete! Time elapsed: {time.time() - start_time:.2f}s")

    # 3. Generalization Testing and Visualization (Hardcore Testing for Long Sequences)
    print("Simulating long-term trajectory with trained Neural ODE...")
    func.eval()
    with torch.no_grad():
        # Provide the model with only a single initial point, and let it integrate forward in time to generate a trajectory spanning 1,500 steps.
        # For an MLP, this is an absolutely impossible task.
        test_steps = 150
        t_test = t_data[:test_steps]
        y0_test = y_data[0] # Select the absolute starting point.
        
        # Predict the entire trajectory.
        pred_trajectory = odeint(func, y0_test, t_test).numpy()
        true_trajectory = y_data[:test_steps].numpy()

    # 4. Drawing Comparison
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], 
            label='Ground Truth', color='blue', alpha=0.6, lw=1)
    # Neural ODE 
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
