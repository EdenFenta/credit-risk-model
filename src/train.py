import pandas as pd
import numpy as np
from pathlib import Path
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import mlflow
import mlflow.sklearn

# Configuration
DATA_PATH = Path("data/processed/customer_with_target.csv")
EXPERIMENT_NAME = "Credit Risk Modeling"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["recency_days", "frequency", "monetary"]
TARGET = "is_high_risk"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility Functions
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def evaluate_model(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

# Main Training Pipeline
def main() -> None:
    logger.info("Loading training data")
    df = load_data()

    X = df[NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Log transform monetary features
    log_transformer = FunctionTransformer(np.log1p, validate=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("log_scale", log_transformer, ["monetary"]),
            ("scale", StandardScaler(), ["recency_days", "frequency"]),
        ],
        remainder="drop",
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, model in models.items():
        logger.info("Training %s", model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_prob)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            # Input example for signature inference
            input_example = X_train.head(5)

            mlflow.sklearn.log_model(
                pipeline,
                name=model_name,
                input_example=input_example,
            )

            logger.info("%s metrics: %s", model_name, metrics)


if __name__ == "__main__":
    main()