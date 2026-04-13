from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_SAMPLING_STRATEGY = "auto"
N_ESTIMATORS = 300
MAX_DEPTH = 12
MIN_SAMPLES_SPLIT = 4
MIN_SAMPLES_LEAF = 2


def load_dataset(data_path: Path) -> pd.DataFrame:
    file_path = Path(data_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Place creditcard.csv in the project's data folder."
        )
    df = pd.read_csv(file_path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the fraud dataset.")
    return df


def remove_outliers_from_nonfraud(df: pd.DataFrame, column: str = "Amount") -> pd.DataFrame:
    if column not in df.columns:
        return df
    non_fraud = df[df["Class"] == 0].copy()
    fraud = df[df["Class"] == 1].copy()

    q1 = non_fraud[column].quantile(0.25)
    q3 = non_fraud[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered_non_fraud = non_fraud[(non_fraud[column] >= lower_bound) & (non_fraud[column] <= upper_bound)]
    cleaned_df = pd.concat([filtered_non_fraud, fraud], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    return cleaned_df


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [col for col in ["Time", "Amount"] if col in feature_columns]
    remaining_features = [col for col in feature_columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("scaled_numeric", numeric_pipeline, numeric_features))

    if remaining_features:
        passthrough_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("other_numeric", passthrough_pipeline, remaining_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title("Fraud Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fraud Detection ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(feature_names: list[str], importances: np.ndarray, output_path: Path, top_n: int = 15) -> None:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    print(f"Loading dataset from {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    df = remove_outliers_from_nonfraud(df, column="Amount")

    feature_columns = [col for col in df.columns if col != "Class"]
    x = df[feature_columns]
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(feature_columns)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
    )
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_processed, y_train)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    model.fit(x_train_balanced, y_train_balanced)

    y_pred = model.predict(x_test_processed)
    y_prob = model.predict_proba(x_test_processed)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    feature_names = preprocessor.get_feature_names_out().tolist()

    plot_confusion_matrix(y_test.to_numpy(), y_pred, OUTPUT_DIR / "fraud_confusion_matrix.png")
    plot_roc_curve(y_test.to_numpy(), y_prob, OUTPUT_DIR / "fraud_roc_curve.png")
    plot_feature_importance(feature_names, model.feature_importances_, OUTPUT_DIR / "fraud_feature_importance.png")

    joblib.dump(model, MODEL_DIR / "fraud_random_forest.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "fraud_preprocessor.joblib")

    metrics = {
        "roc_auc": float(roc_auc),
        "precision_fraud": float(report["1"]["precision"]),
        "recall_fraud": float(report["1"]["recall"]),
        "f1_fraud": float(report["1"]["f1-score"]),
        "support_fraud": int(report["1"]["support"]),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "balanced_train_rows": int(len(x_train_balanced)),
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "fraud_metrics.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / "fraud_classification_report.csv")

    print("Training complete.")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Model saved to: {MODEL_DIR / 'fraud_random_forest.joblib'}")
    print(f"Confusion matrix plot: {OUTPUT_DIR / 'fraud_confusion_matrix.png'}")
    print(f"ROC curve plot:        {OUTPUT_DIR / 'fraud_roc_curve.png'}")


if __name__ == "__main__":
    main()
