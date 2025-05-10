import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def main():
    input_dir = "train_preprocessed"
    data_frames = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, filename))
            data_frames.append(df)
    
    data = pd.concat(data_frames, ignore_index=True)
    X = data[['Day']]
    y = data['Temperature_scaled']
    
    model = LinearRegression()
    model.fit(X, y)
    
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, os.path.join("model", "linear_model.pkl"))
    print("Модель обучена и сохранена в 'model/linear_model.pkl'")

if __name__ == "__main__":
    main()
