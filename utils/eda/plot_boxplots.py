import matplotlib.pyplot as PLT


def plot_boxplots(dataset, continuous_columns, save_path):
  n_cols = 3
  n_rows = (len(continuous_columns) + n_cols - 1) // n_cols

  fig, axes = PLT.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
  axes = axes.flatten() if n_rows > 1 else axes

  for i, col in enumerate(continuous_columns):
    if i < len(axes):
      axes[i].boxplot(dataset[col])
      axes[i].set_title(f'Boxplot of {col}')
      axes[i].set_ylabel(col)
      axes[i].grid(True, alpha=0.3)

  # Hide empty subplots
  for i in range(len(continuous_columns), len(axes)):
    axes[i].set_visible(False)

  PLT.tight_layout()
  PLT.savefig(save_path + 'boxplots.png', dpi=300, bbox_inches='tight')
  PLT.close()
  print("Boxplots saved as 'boxplots.png'")
