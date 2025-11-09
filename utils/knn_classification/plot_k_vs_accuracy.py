import matplotlib.pyplot as PLT
from utils import draw_line


def plot_k_vs_accuracy(k_values, accuracies, save_path):
  draw_line("Plotting K values and Accuracies")

  PLT.figure(figsize=(10, 6))
  PLT.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
  PLT.xlabel('k Value')
  PLT.ylabel('Accuracy')
  PLT.title('KNN: k Value vs Accuracy')
  PLT.grid(True, alpha=0.3)
  PLT.xticks(k_values)

  # Save to file
  PLT.savefig(save_path, dpi=300, bbox_inches='tight')
  PLT.close()  # Close the figure to free memory

  print(f"Plot saved to {save_path}")
