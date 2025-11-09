import pandas as PD
import numpy as NP
from collections import Counter


def knn_predict(X_train, y_train, X_test, k=5):
  # KNN prediction function
  predictions = []
  X_train_values = X_train.values if isinstance(
      X_train, PD.DataFrame) else X_train
  y_train_values = y_train.values if isinstance(
      y_train, PD.Series) else y_train
  X_test_values = X_test.values if isinstance(
      X_test, PD.DataFrame) else X_test

  for test_point in X_test_values:
    # Calculate distances to all training points
    distances = []
    for i, train_point in enumerate(X_train_values):
      distance = NP.sqrt(NP.sum((test_point - train_point) ** 2))
      distances.append((distance, y_train_values[i]))

    # Get k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for dist, label in distances[:k]]

    # Majority vote
    most_common = Counter(k_nearest).most_common(1)[0][0]
    predictions.append(most_common)

  return NP.array(predictions)
