import numpy as NP
from .knn_accuracy import knn_accuracy
from utils import draw_line


def find_optimal_k(X_train, X_test, y_train, y_test, max_k=15):
  draw_line("Finding optimal K value")
  print("Testing different k values to find best accuracy:")

  k_values = range(3, max_k + 1, 2)  # Only odd numbers to avoid ties
  accuracies = []

  for k in k_values:
    accuracy = knn_accuracy(X_train, y_train, X_test, y_test, k)
    accuracies.append(accuracy)
    print(f"k={k}: Accuracy = {accuracy:.4f}")

  # Find best k
  best_idx = NP.argmax(accuracies)
  best_k = k_values[best_idx]
  best_accuracy = accuracies[best_idx]

  print(f"\nOptimal k: {best_k} with accuracy: {best_accuracy:.4f}")

  return best_k, k_values, accuracies
