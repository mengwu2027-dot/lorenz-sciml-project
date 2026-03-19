# Models

This directory contains the neural network model definitions used in this project for learning the dynamics of the Lorenz system.

- `mlp_model.py`: Implements a baseline Multi-Layer Perceptron (MLP) that learns a discrete-time mapping of the system state.
- `neural_ode.py`: Defines the Neural ODE model, where a neural network parameterizes the continuous-time dynamics.
- `pinn.py`: Implements the Physics-Informed Neural Network (PINN), incorporating the Lorenz system equations into the loss function.

These models are used and trained via the scripts located in the `src/models/` directory.
