import numpy as np
from scipy.integrate import solve_ivp
from lorenz_solver import lorenz

t_span = (0, 40)
t_eval = np.linspace(0, 40, 10000)

initial_state = [1.0, 1.0, 1.0]

sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

data = sol.y.T

np.save("../data/lorenz_dataset.npy", data)

print("Dataset saved.")
