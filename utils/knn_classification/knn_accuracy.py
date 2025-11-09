import numpy as NP
import pandas as PD
from .knn_predict import knn_predict


def knn_accuracy(X_train, y_train, X_test, y_test, k=5):
  # Calculate KNN accuracy
  y_pred = knn_predict(X_train, y_train, X_test, k)
  y_true = y_test.values if isinstance(y_test, PD.Series) else y_test
  return NP.mean(y_pred == y_true)
