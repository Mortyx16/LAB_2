import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    input_dir = "train"
    output_dir = "train_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    
    all_temperature = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, filename))
            all_temperature.append(df['Temperature'])
    
    combined_temps = pd.concat(all_temperature).values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(combined_temps)
    
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, os.path.join("model", "scaler.pkl"))
    print("Scaler сохранён в 'model/scaler.pkl'")
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, filename))
            df['Temperature_scaled'] = scaler.transform(df[['Temperature']])
            output_filename = os.path.join(output_dir, filename)
            df.to_csv(output_filename, index=False)
            print(f"Преобразованный файл сохранён: {output_filename}")

if __name__ == "__main__":
    main()
