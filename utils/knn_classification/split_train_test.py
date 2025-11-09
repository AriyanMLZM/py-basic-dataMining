import numpy as NP
from .. import draw_line


def split_train_test(dataset, target_column, test_size=0.2, random_state=42):
  draw_line('Spliting data into training and testing sets')

  # Separate features and target
  X = dataset.drop(columns=[target_column])
  y = dataset[target_column]

  # Manual train-test split
  NP.random.seed(random_state)
  n_samples = len(dataset)
  n_test = int(n_samples * test_size)

  # Shuffle indices
  indices = NP.random.permutation(n_samples)
  test_indices = indices[:n_test]
  train_indices = indices[n_test:]

  X_train = X.iloc[train_indices]
  X_test = X.iloc[test_indices]
  y_train = y.iloc[train_indices]
  y_test = y.iloc[test_indices]

  print(
      f"Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
  print(f"Testing set: {X_test.shape[0]} samples ({(test_size)*100:.0f}%)")
  print(f"Total features: {X_train.shape[1]}")

  return X_train, X_test, y_train, y_test
