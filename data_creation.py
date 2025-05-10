import os
import numpy as np
import pandas as pd

def generate_data(num_days=30, anomaly_probability=0.1, noise_std=2.0):
    days = np.arange(1, num_days + 1)
    temp = 20 + 10 * np.sin(2 * np.pi * days / num_days)
    noise = np.random.normal(0, noise_std, size=num_days)
    temp += noise
    # Введение аномалий
    for i in range(num_days):
        if np.random.rand() < anomaly_probability:
            temp[i] += np.random.choice([15, -15])
    return pd.DataFrame({'Day': days, 'Temperature': temp})

def main():
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    num_train_files = 5
    num_test_files = 2

    for i in range(num_train_files):
        df = generate_data()
        filename = os.path.join("train", f"data_train_{i+1}.csv")
        df.to_csv(filename, index=False)
        print(f"Создан файл: {filename}")
        
    for i in range(num_test_files):
        df = generate_data()
        filename = os.path.join("test", f"data_test_{i+1}.csv")
        df.to_csv(filename, index=False)
        print(f"Создан файл: {filename}")

if __name__ == "__main__":
    main()
