import matplotlib.pyplot as PLT


def plot_correlation_matrix(dataset, numerical_columns, save_path):
  corr_matrix = dataset[numerical_columns].corr()

  PLT.figure(figsize=(12, 10))
  im = PLT.imshow(corr_matrix, cmap='coolwarm',
                  aspect='auto', vmin=-1, vmax=1)

  cbar = PLT.colorbar(im, fraction=0.046, pad=0.04)
  cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

  PLT.xticks(range(len(numerical_columns)),
             numerical_columns, rotation=45, ha='right')
  PLT.yticks(range(len(numerical_columns)), numerical_columns)

  # Add correlation values as text
  for i in range(len(numerical_columns)):
    for j in range(len(numerical_columns)):
      PLT.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

  PLT.title('Correlation Matrix of Numerical Features',
            fontsize=14, fontweight='bold')
  PLT.tight_layout()
  PLT.savefig(save_path + 'correlation_matrix.png',
              dpi=300, bbox_inches='tight')
  PLT.close()
  print("Correlation matrix saved as 'correlation_matrix.png'")

  # Print strong correlations
  print("\nStrong Correlations (|r| > 0.5):")
  strong_corrs = []
  for i in range(len(numerical_columns)):
    for j in range(i+1, len(numerical_columns)):
      corr = corr_matrix.iloc[i, j]
      if abs(corr) > 0.5:
        strong_corrs.append(
            (numerical_columns[i], numerical_columns[j], corr))

  for feat1, feat2, corr in strong_corrs:
    print(f"  {feat1} - {feat2}: {corr:.3f}")
