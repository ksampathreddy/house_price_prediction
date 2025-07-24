import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

def train_and_save_model():
    # Load dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target

    # Features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R²: {train_score:.3f}")
    print(f"Test R²: {test_score:.3f}")

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_and_save_model()