# src/data/preprocess.py

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from configs import settings

def preprocess_data(X, y):
    """
    Scale features and split into train/test sets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.

    Returns
    -------
    X_train : ndarray
        Scaled training features.
    X_test : ndarray
        Scaled test features.
    y_train : ndarray
        Training targets.
    y_test : ndarray
        Test targets.
    scaler : StandardScaler
        Fitted scaler (for later use on new data).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test, scaler
