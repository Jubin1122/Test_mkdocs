# src/main.py

from src.data.load_data import load_data, split_data
from src.data.preprocess import preprocess_data
from src.models.train import train_model, evaluate_model, save_model

def main() -> None:
    """
    Run the end-to-end ML pipeline:

    1. Load CSV data
    2. Split into X/y
    3. Scale and split train/test
    4. Train RandomForest
    5. Evaluate on test set
    6. Save model to disk
    """
    print("Starting ML pipeline...")

    print("Loading data...")
    df = load_data()
    X, y = split_data(df)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)

    print("Training model...")
    model = train_model(X_train, y_train, n_estimators=150)

    print("Evaluating model...")
    accuracy, report = evaluate_model(model, X_test, y_test)

    save_model(model)
    print("Pipeline completed!")


if __name__ == "__main__":
    main()
