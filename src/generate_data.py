import numpy as np
import os
from lorenz_solver import generate_lorenz_data

def main():
    print("Starting to generate Lorenz system data...")
    # Call lorenz_solver.py to generate data.
    t, data = generate_lorenz_data()
    
    # Locate the `data` folder in the project root directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save time t and trajectory data as an .npy file.
    save_path = os.path.join(data_dir, 'lorenz_ground_truth.npy')
    np.save(save_path, {'t': t, 'data': data})
    
    print(f"Data generation complete! Saved to: {save_path}")
    print(f"Data shape: {data.shape}")

if __name__ == "__main__":
    main()
