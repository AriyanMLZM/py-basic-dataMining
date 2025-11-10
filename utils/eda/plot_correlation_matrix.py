import matplotlib.pyplot as PLT
from .. import draw_line


def plot_correlation_matrix(dataset, corr_columns, target_column, save_path):
  draw_line("Correlation Matrix")

  # Add target at end
  corr_columns.remove(target_column)
  corr_columns.append(target_column)

  corr_matrix = dataset[corr_columns].corr()

  PLT.figure(figsize=(14, 12))
  im = PLT.imshow(corr_matrix, cmap='coolwarm',
                  aspect='auto', vmin=-1, vmax=1)

  cbar = PLT.colorbar(im, fraction=0.046, pad=0.04)
  cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)

  PLT.xticks(range(len(corr_columns)), corr_columns,
             rotation=45, ha='right', fontsize=8)
  PLT.yticks(range(len(corr_columns)), corr_columns, fontsize=8)

  # Add correlation values as text (only for notable correlations)
  for i in range(len(corr_columns)):
    for j in range(len(corr_columns)):
      corr_val = corr_matrix.iloc[i, j]
      PLT.text(j, i, f'{corr_val:.2f}', ha='center', va='center', fontsize=7, bbox=dict(
          boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

  PLT.title('Correlation Matrix After One-Hot Encoding',
            fontsize=14, fontweight='bold')
  PLT.tight_layout()
  PLT.savefig(save_path + 'correlation_matrix.png',
              dpi=300, bbox_inches='tight')
  PLT.close()
  print("Correlation matrix saved as 'correlation_matrix.png'")

  # Print correlations with target
  print("\nFeature Correlations with Target:")
  target_corrs = []
  for col in corr_columns:
    if col != target_column:
      corr = corr_matrix.loc[col, target_column]
      target_corrs.append((col, corr))

  # Sort by absolute correlation strength
  target_corrs.sort(key=lambda x: abs(x[1]), reverse=True)

  for col, corr in target_corrs:
    strength = "STRONG" if abs(
        corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
    direction = "positive" if corr > 0 else "negative"
    print(f"  {col}: {corr:.3f} ({strength} {direction})")
