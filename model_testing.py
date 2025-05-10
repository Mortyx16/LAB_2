import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

def main():
    scaler = joblib.load(os.path.join("model", "scaler.pkl"))
    model = joblib.load(os.path.join("model", "linear_model.pkl"))
    
    test_dir = "test"
    data_frames = []
    
    for filename in os.listdir(test_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(test_dir, filename))
            df['Temperature_scaled'] = scaler.transform(df[['Temperature']])
            data_frames.append(df)
    
    test_data = pd.concat(data_frames, ignore_index=True)
    X_test = test_data[['Day']]
    y_test = test_data['Temperature_scaled']
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error на тестовых данных:", mse)

if __name__ == "__main__":
    main()
