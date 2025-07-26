# src/models/train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from configs import settings

def train_model(X_train, y_train, n_estimators: int = 100) -> RandomForestClassifier:
    """
    Train a RandomForest classifier.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training features.
    y_train : array-like of shape (n_samples,)
        Training targets.
    n_estimators : int, default=100
        Number of trees in the forest.

    Returns
    -------
    model : RandomForestClassifier
        Fitted classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=settings.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Compute accuracy and print a full classification report.

    Parameters
    ----------
    model : sklearn-like estimator
        A trained classifier with a `.predict` method.
    X_test : array-like of shape (n_samples, n_features)
        Test features.
    y_test : array-like of shape (n_samples,)
        True test targets.

    Returns
    -------
    accuracy : float
        Fraction of correct predictions.
    report : str
        Text report showing precision, recall, f1-score by class.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    return accuracy, report


def save_model(model, path: str = settings.MODEL_SAVE_PATH) -> None:
    """
    Persist a trained model to disk.

    Parameters
    ----------
    model : sklearn-like estimator
        The model to save.
    path : str
        Path (including filename) where the `.pkl` will be written.

    Returns
    -------
    None
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")
