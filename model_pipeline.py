"""
model_pipeline.py
Modular ML pipeline for house price prediction (King County dataset).

Functions:
    prepare_data()   -> load + preprocess + split the data
    train_model()    -> train a Gradient Boosting Regressor
    evaluate_model() -> compute R2, MAE, RMSE on the test set
    save_model()     -> save the trained model with joblib
    load_model()     -> load a saved model
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "price"
COLUMNS_TO_DROP = ["id"]  # identifier, not a feature


def prepare_data(csv_path, test_size=0.1, random_state=2):
    """Load the CSV, clean it and split it into train/test sets.

    Args:
        csv_path (str): path to kc_house_data.csv.
        test_size (float): fraction of rows used for the test set.
        random_state (int): seed so the split is identical on every run.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Cleaning: drop useless columns, duplicates and missing values
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    df = df.drop_duplicates().dropna()

    # Same idea as the notebook: 1 if the house was sold in 2014, else 0.
    # (dates look like "20141013T000000", so we read the first 4 characters;
    # the notebook compared the whole string to 2014, which is always False.)
    df["date"] = (df["date"].astype(str).str[:4] == "2014").astype(int)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from the data")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train,
    y_train,
    n_estimators=400,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=2,
    random_state=2,
):
    """Train a Gradient Boosting Regressor.

    Args:
        X_train (pd.DataFrame): training features.
        y_train (pd.Series): training target (price).
        n_estimators (int): number of boosting stages.
        max_depth (int): depth of each tree.
        learning_rate (float): shrinkage applied to each tree.
        min_samples_split (int): min samples needed to split a node.
        random_state (int): seed for reproducibility.

    Returns:
        GradientBoostingRegressor: the fitted model.
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        loss="squared_error",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a regression model on the test set.

    Args:
        model: a fitted scikit-learn regressor.
        X_test (pd.DataFrame): test features.
        y_test (pd.Series): true prices.

    Returns:
        dict: {"r2": float, "mae": float, "rmse": float}
    """
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def save_model(model, path="model.joblib"):
    """Save a trained model to disk with joblib.

    Args:
        model: the fitted model.
        path (str): destination file.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path="model.joblib"):
    """Load a model previously saved with save_model().

    Args:
        path (str): path of the saved model.

    Returns:
        The loaded model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model at {path}. Run: python main.py --train"
        )
    return joblib.load(path)
