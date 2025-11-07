from .. import draw_line


def normalize_data(dataset, continuous_columns, method='minmax'):
  draw_line(f"Normalizing Data - Method: {method.upper()}")

  df_normalized = dataset.copy()
  scaler_params = {}

  if continuous_columns:
    print("Continuous columns (normalized):")
    for col in continuous_columns:
      if method == 'minmax':
        min_val = dataset[col].min()
        max_val = dataset[col].max()
        if max_val != min_val:
          df_normalized[col] = ((
              dataset[col] - min_val) / (max_val - min_val)).round(4)
          scaler_params[col] = {'min': min_val, 'max': max_val}
          print(
              f"  {col}: Min-Max normalized [{min_val:.2f}, {max_val:.2f}] → [0, 1]")
        else:
          df_normalized[col] = 0
          print(f"  {col}: Constant value, set to 0")

      elif method == 'zscore':
        mean_val = dataset[col].mean()
        std_val = dataset[col].std()
        if std_val != 0:
          df_normalized[col] = ((dataset[col] - mean_val) / std_val).round(4)
          scaler_params[col] = {'mean': mean_val, 'std': std_val}
          print(
              f"  {col}: Z-score normalized (mean: {mean_val:.2f}, std: {std_val:.2f})")
        else:
          df_normalized[col] = 0
          print(f"  {col}: Constant value, set to 0")

  return df_normalized
