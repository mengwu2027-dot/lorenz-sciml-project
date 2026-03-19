import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Import the pre-written PINN model and physics-based loss function.
from models.pinn import PINN, physics_loss

def main():
    # 1. Path Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'lorenz_ground_truth.npy')
    
    # Loading Real Data
    data_dict = np.load(data_path, allow_pickle=True).item()
    
    # Fitting chaotic systems over long time spans using PINNs is extremely difficult.
    # The standard practice in the academic community is to first demonstrate the exceptionally high fitting accuracy of PINNs within a relatively short time window.
    # Here, we use the trajectory of the first 500 time steps (i.e., 5 seconds) as the training domain.
   
    
    train_steps = 200
    t_data = torch.tensor(data_dict['t'][:train_steps], dtype=torch.float32).view(-1, 1)
    xyz_data = torch.tensor(data_dict['data'][:train_steps, :], dtype=torch.float32)
    
    # Perform data standardization calculations.
    xyz_mean = xyz_data.mean(dim=0)
    xyz_std = xyz_data.std(dim=0)
    # The target the network aims to fit becomes the standardized data (with a mean of 0 and a variance of 1).
    xyz_data_normalized = (xyz_data - xyz_mean) / xyz_std
    
    # Initialize Model
    model = PINN(hidden_dim=64, num_layers=6) 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    epochs = 10000 
    lambda_physics = 1e-3 # With the addition of standardization, 1e-3 will no longer trigger an explosion.

    print("Starting PINN training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # (1) Data Loss: Have the model fit the standardized coordinate points.
        xyz_pred_normalized = model(t_data)
        loss_data = mse_loss(xyz_pred_normalized, xyz_data_normalized)
        
        # (2) Physics Loss: Core Modifications
        t_physics = t_data.clone().requires_grad_(True)
        xyz_pred_norm_phys = model(t_physics)
        
        # The standardized data predicted by the neural network must be reverse-converted back into actual physical quantities before they can be substituted into physical equations.
        x_real = xyz_pred_norm_phys[:, 0:1] * xyz_std[0] + xyz_mean[0]
        y_real = xyz_pred_norm_phys[:, 1:2] * xyz_std[1] + xyz_mean[1]
        z_real = xyz_pred_norm_phys[:, 2:3] * xyz_std[2] + xyz_mean[2]
        
        # Computing Derivatives Using Automatic Differentiation (Differentiating Real Physical Quantities)
        dx_dt = torch.autograd.grad(x_real, t_physics, grad_outputs=torch.ones_like(x_real), create_graph=True)[0]
        dy_dt = torch.autograd.grad(y_real, t_physics, grad_outputs=torch.ones_like(y_real), create_graph=True)[0]
        dz_dt = torch.autograd.grad(z_real, t_physics, grad_outputs=torch.ones_like(z_real), create_graph=True)[0]
        
        # Substitute into the Lorenz equations.
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        f_x = dx_dt - sigma * (y_real - x_real)
        f_y = dy_dt - (x_real * (rho - z_real) - y_real)
        f_z = dz_dt - (x_real * y_real - beta * z_real)
        
        loss_phys = torch.mean(f_x**2 + f_y**2 + f_z**2)
        
        # Combined Loss
        loss = loss_data + lambda_physics * loss_phys
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {loss.item():.4f} "
                  f"(Data: {loss_data.item():.4f}, Phys: {loss_phys.item():.4f})")
            
    print("Training complete!")

   # 3. Generalization Testing and Visualization (Including Extrapolation Prediction)
    print("Generating extrapolation prediction plot...")
    model.eval()
    with torch.no_grad():
        # Increase the test step count to 400 (comprising 200 known steps + 200 unknown steps).
        test_steps = 400 
        t_test = torch.tensor(data_dict['t'][:test_steps], dtype=torch.float32).view(-1, 1)
        
        # Network Prediction and Denormalization
        predicted_norm = model(t_test)
        predicted_xyz = (predicted_norm * xyz_std + xyz_mean).numpy()
        
        true_xyz = data_dict['data'][:test_steps, :]
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the actual blue line for the first 400 steps.
    ax.plot(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2], 
            label='Ground Truth (400 steps)', color='blue', alpha=0.4, lw=1)
            
    # Distinguishing Between the Training Zone and the Prediction Zone
    # 1. The trained region (first 200 steps) is indicated by a green dashed line.
    ax.plot(predicted_xyz[:200, 0], predicted_xyz[:200, 1], predicted_xyz[:200, 2], 
            label='PINN (Training Region)', color='green', lw=2, linestyle='--')
            
    # 2. The unknown future region (200 to 400 steps) illustrates its extrapolation capability using a solid red line.
    ax.plot(predicted_xyz[200:, 0], predicted_xyz[200:, 1], predicted_xyz[200:, 2], 
            label='PINN (Future Prediction)', color='red', lw=2)
    
    # Draw a point to mark the "present" moment (i.e., the boundary where the data runs out).
    ax.scatter(*predicted_xyz[199, :], color='black', s=50, label='Prediction Start', zorder=5)

    ax.set_title("Lorenz Attractor: Ground Truth vs PINN")
    ax.legend()
    
    # Save Image
    figures_dir = os.path.join(current_dir, '..', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, 'pinn_prediction.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {save_path}")

if __name__ == "__main__":
    main()
    
