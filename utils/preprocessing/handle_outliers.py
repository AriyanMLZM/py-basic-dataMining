import numpy as NP
from .. import draw_line


def handle_outliers(dataset, continuous_columns, method='cap'):
  draw_line(f"Handling Outliers - Method: {method.upper()}")

  if not continuous_columns:
    print("No continuous columns for outlier handling.")
    return dataset.copy()

  dataset_clean = dataset.copy()
  total_outliers_removed = 0

  for col in continuous_columns:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_count_before = len(
        dataset[(dataset[col] < lower_bound) | (dataset[col] > upper_bound)])

    if method == 'cap':
      # Cap outliers at bounds
      dataset_clean[col] = NP.where(
          dataset_clean[col] < lower_bound, lower_bound, dataset_clean[col])
      dataset_clean[col] = NP.where(
          dataset_clean[col] > upper_bound, upper_bound, dataset_clean[col])
      print(f"{col}: Capped {outlier_count_before} outliers")

    elif method == 'remove':
      # Remove rows with outliers
      outlier_mask = (dataset_clean[col] < lower_bound) | (
          dataset_clean[col] > upper_bound)
      rows_removed = outlier_mask.sum()
      dataset_clean = dataset_clean[~outlier_mask]
      total_outliers_removed += rows_removed
      print(f"{col}: Removed {rows_removed} outlier rows")

    elif method == 'ignore':
      print(f"{col}: Keeping {outlier_count_before} outliers (no action)")

  if method == 'remove':
    print(
        f"\nTotal rows removed due to outliers: {total_outliers_removed}")

  print(f"Shape after outlier handling: {dataset_clean.shape}")
  return dataset_clean
