import numpy as np
from utils import draw_line


def evaluate_model(y_true, y_pred):
  draw_line("Evaluating model performance")

  accuracy = np.mean(y_true == y_pred)

  # Confusion matrix
  cm = np.zeros((2, 2))
  for true, pred in zip(y_true, y_pred):
    cm[int(true), int(pred)] += 1

  # Calculate metrics
  tp = cm[1, 1]
  tn = cm[0, 0]
  fp = cm[0, 1]
  fn = cm[1, 0]

  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  f1 = 2 * (precision * recall) / (precision +
                                   recall) if (precision + recall) > 0 else 0

  print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1-Score: {f1:.4f}")
  print(f"\nConfusion Matrix:")
  print(f"[TN: {tn:.0f}, FP: {fp:.0f}]")
  print(f"[FN: {fn:.0f}, TP: {tp:.0f}]")

  return accuracy, precision, recall, f1
