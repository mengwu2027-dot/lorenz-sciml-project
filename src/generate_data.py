import numpy as np
import os
from lorenz_solver import generate_lorenz_data

def main():
    print("Starting to generate Lorenz system data...")
    # 调用你刚才写的 solver 生成数据
    t, data = generate_lorenz_data()
    
    # 找到项目根目录下的 data 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 将时间 t 和轨迹 data 保存为一个 .npy 文件
    save_path = os.path.join(data_dir, 'lorenz_ground_truth.npy')
    np.save(save_path, {'t': t, 'data': data})
    
    print(f"Data generation complete! Saved to: {save_path}")
    print(f"Data shape: {data.shape}")

if __name__ == "__main__":
    main()
