import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz_dynamics(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """
    定义 Lorenz 系统的微分方程。
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_data(t_span=(0, 20), dt=0.01, initial_state=[-8.0, 7.0, 27.0]):
    """
    使用 RK45 (Runge-Kutta) 方法生成 Lorenz 系统的真实轨迹数据。
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    # 使用 scipy 的 solve_ivp 进行高精度数值求解
    solution = solve_ivp(
        fun=lorenz_dynamics,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8, # 设置较高的容差以确保 Ground Truth 的准确性
        atol=1e-8
    )
    
    return solution.t, solution.y.T # 返回形状为 (N,) 和 (N, 3) 的数组

# 测试代码
if __name__ == "__main__":
    t, data = generate_lorenz_data()
    print(f"Generated data shape: {data.shape}")
    
    # 快速画一个 3D 图验证一下蝴蝶形状
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=0.5, color='b')
    ax.set_title("Lorenz Attractor (Ground Truth)")
    plt.show()
