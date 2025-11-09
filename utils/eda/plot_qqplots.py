import matplotlib.pyplot as PLT
import numpy as NP


def plot_qqplots(dataset, continuous_columns, save_path):
  n_cols = 3
  n_rows = (len(continuous_columns) + n_cols - 1) // n_cols

  fig, axes = PLT.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
  axes = axes.flatten() if n_rows > 1 else axes

  for i, col in enumerate(continuous_columns):
    if i < len(axes):
      # Manual Q-Q plot implementation
      data = dataset[col].dropna().sort_values()
      n = len(data)
      # Theoretical quantiles from normal distribution
      theoretical_quantiles = NP.sort(NP.random.normal(0, 1, n))
      # Standardize the data
      standardized_data = (data - data.mean()) / data.std()

      axes[i].scatter(theoretical_quantiles,
                      standardized_data, alpha=0.6)

      # Add reference line (y = x)
      min_val = min(theoretical_quantiles.min(), standardized_data.min())
      max_val = max(theoretical_quantiles.max(), standardized_data.max())
      axes[i].plot([min_val, max_val], [
                   min_val, max_val], 'r--', alpha=0.8)

      axes[i].set_title(f'Q-Q Plot of {col}')
      axes[i].set_xlabel('Theoretical Quantiles')
      axes[i].set_ylabel('Sample Quantiles')
      axes[i].grid(True, alpha=0.3)

  # Hide empty subplots
  for i in range(len(continuous_columns), len(axes)):
    axes[i].set_visible(False)

  PLT.tight_layout()
  PLT.savefig(save_path + 'qq_plots.png', dpi=300, bbox_inches='tight')
  PLT.close()
  print("Q-Q plots saved as 'qq_plots.png'")
