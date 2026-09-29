import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train_simple_models():
    # Load dataset
    df = pd.read_csv("simple_turbine_data.csv")

    features = ["temperature", "pressure", "vibration"]
    X = df[features]
    y_status = df["status"]
    y_rul = df["rul_cycles"]

    # 1. Status Classifier Model
    encoder = LabelEncoder()
    y_status_encoded = encoder.fit_transform(y_status)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y_status_encoded)

    # 2. RUL Regressor Model
    reg_rul = RandomForestRegressor(n_estimators=50, random_state=42)
    reg_rul.fit(X, y_rul)

    # Save artifacts into a clean dictionary
    with open("simple_turbine_model.pkl", "wb") as f:
        pickle.dump({
            "status_model": clf,
            "rul_model": reg_rul,
            "encoder": encoder,
            "features": features
        }, f)

    print("✓ Simple models successfully trained and saved as 'simple_turbine_model.pkl'")

if __name__ == "__main__":
    train_simple_models()