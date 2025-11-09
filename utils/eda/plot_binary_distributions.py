import matplotlib.pyplot as PLT


def plot_binary_distributions(dataset, binary_columns, save_path):
  n_cols = 3
  n_rows = (len(binary_columns) + n_cols - 1) // n_cols

  fig, axes = PLT.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
  axes = axes.flatten() if n_rows > 1 else axes

  for i, col in enumerate(binary_columns):
    if i < len(axes):
      value_counts = dataset[col].value_counts().sort_index()
      bars = axes[i].bar(value_counts.index, value_counts.values,
                         color=['lightcoral', 'lightblue'], alpha=0.7, edgecolor='black')
      axes[i].set_title(f'Distribution of {col}')
      axes[i].set_xlabel(col)
      axes[i].set_ylabel('Count')
      axes[i].set_xticks(value_counts.index)
      axes[i].grid(True, alpha=0.3)

      # Add value labels on bars
      for bar in bars:
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                     f'{height}\n({height/len(dataset)*100:.1f}%)',
                     ha='center', va='bottom')

  # Hide empty subplots
  for i in range(len(binary_columns), len(axes)):
    axes[i].set_visible(False)

  PLT.tight_layout()
  PLT.savefig(save_path + 'binary_distributions.png',
              dpi=300, bbox_inches='tight')
  PLT.close()
  print("Binary distributions saved as 'binary_distributions.png'")
