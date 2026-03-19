import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz_dynamics(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    Define the differential equations of the Lorenz system.
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(t_span=(0, 20), dt=0.01, initial_state=[-8.0, 7.0, 27.0]):
    """
    Generate true trajectory data for the Lorenz system using the RK45 (Runge-Kutta) method.
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    # Performing High-Precision Numerical Solutions using SciPy's `solve_ivp`
    solution = solve_ivp(
        fun=lorenz_dynamics,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8, # Set a higher tolerance to ensure the accuracy of the Ground Truth.
        atol=1e-8
    )
    
    return solution.t, solution.y.T # Returns arrays with shapes (N,) and (N, 3).

# Test Code
if __name__ == "__main__":
    t, data = generate_lorenz_data()
    print(f"Generated data shape: {data.shape}")
    
    # Let's quickly sketch a 3D diagram to verify the butterfly shape.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5, color='b')
    ax.set_title("Lorenz Attractor (Ground Truth)")
    plt.show()
