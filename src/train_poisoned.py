import argparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# ==============================
# DATA POISONING HELPERS
# ==============================

def poison_features(X, poison_rate, seed=42):
    """
    Replace a fraction of feature rows with random values
    drawn from realistic ranges.
    """
    rng = np.random.RandomState(seed)
    Xp = X.copy()
    n = len(Xp)
    k = int(poison_rate * n)

    if k == 0:
        return Xp

    idx = rng.choice(n, size=k, replace=False)

    mins = Xp.min(axis=0)
    maxs = Xp.max(axis=0)

    for i in idx:
        Xp.iloc[i] = rng.uniform(mins, maxs)

    return Xp


def poison_labels(y, poison_rate, seed=42):
    """
    Randomly flip labels for a subset of the training set.
    """
    rng = np.random.RandomState(seed)
    yp = y.copy()
    n = len(yp)
    k = int(poison_rate * n)

    if k == 0:
        return yp

    idx = rng.choice(n, size=k, replace=False)
    classes = np.unique(yp)

    for i in idx:
        yp.iloc[i] = rng.choice(classes[classes != yp.iloc[i]])

    return yp


# ==============================
# TRAINING SCRIPT (with poisoning)
# ==============================

def train_model(model_name, poison_rate, poison_type):
    print(f"Training model: {model_name}")
    print(f"Poison rate: {poison_rate}, Poison type: {poison_type}")

    # 1. Load Data
    data_path = "data/iris.csv"
    df = pd.read_csv(data_path)

    X = df.drop("species", axis=1)
    y_raw = df["species"]

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw))

    # Train-test split (IMPORTANT: Only poison training data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Apply poisoning
    if poison_type == "feature":
        X_train_poisoned = poison_features(X_train, poison_rate)
        y_train_poisoned = y_train

    elif poison_type == "label":
        X_train_poisoned = X_train
        y_train_poisoned = poison_labels(y_train, poison_rate)

    elif poison_type == "both":
        X_train_poisoned = poison_features(X_train, poison_rate)
        y_train_poisoned = poison_labels(y_train, poison_rate)

    else:
        X_train_poisoned = X_train
        y_train_poisoned = y_train

    # 3. Model + Grid Search
    model = LogisticRegression(solver='liblinear', random_state=42)
    param_grid = {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}

    if model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

    # ==============================
    # MLflow Logging
    # ==============================
    with mlflow.start_run(run_name=f"{model_name}_poisoned"):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("poison_rate", poison_rate)
        mlflow.log_param("poison_type", poison_type)

        mlflow.log_params({f"param_{k}": str(v) for k, v in param_grid.items()})

        # Train with CV
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_poisoned, y_train_poisoned)

        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

        # Evaluate on clean test set
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)

        print("=== Results ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Log model
        mlflow.sklearn.log_model(best_model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Poisoned Iris Model")

    parser.add_argument("--model_name", type=str,
                        default="logistic_regression",
                        choices=["logistic_regression", "random_forest"])

    parser.add_argument("--poison_rate", type=float,
                        default=0.0,
                        help="Fraction of training data to poison")

    parser.add_argument("--poison_type", type=str,
                        default="none",
                        choices=["none", "feature", "label", "both"],
                        help="Type of poisoning to apply")


    args = parser.parse_args()

    # Set MLflow tracking URI
    MLFLOW_URI = "http://35.200.211.90:8100"
    mlflow.set_tracking_uri(MLFLOW_URI)

    
    experiment_name = "Iris_Data_Poisoning_Experiment"

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)

    train_model(args.model_name, args.poison_rate, args.poison_type)
    print("Done")