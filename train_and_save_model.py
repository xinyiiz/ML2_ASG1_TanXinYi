# train_and_save_model.py
from pathlib import Path
import pandas as pd
import joblib

from src.model import train


def main():
    data_path = Path("data/day_2011.csv")  # adjust if needed
    model_path = Path("model/final_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    # This uses:
    # DateFeatureAdder → ColumnTransformer → RandomForest
    # exactly the same pipeline CI/tests expect
    pipe = train(df)

    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path.resolve()}")


if __name__ == "__main__":
    main()
