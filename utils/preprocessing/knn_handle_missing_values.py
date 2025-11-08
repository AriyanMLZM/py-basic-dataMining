import numpy as NP
from .. import draw_line


def knn_handle_missing_values(dataset, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns, k=5):
  draw_line('Handling Missing Values - KNN')

  dataset_imputed = dataset.copy()
  missing_before = dataset.isnull().sum().sum()

  if missing_before == 0:
    print("No missing values to handle.")
    return dataset_imputed

  # Use continuous columns for distance calculation (they have the most information)
  distance_columns = continuous_columns.copy()

  # Add binary columns to distance calculation if they exist
  if binary_columns:
    distance_columns.extend(binary_columns)

  print(
      f"Using {len(distance_columns)} features for KNN distance calculation: {distance_columns}")

  # First, temporarily fill missing values in distance columns with mean for initial calculation
  temp_dataset = dataset[distance_columns].copy()
  for col in distance_columns:
    if temp_dataset[col].isnull().sum() > 0:
      temp_dataset[col] = temp_dataset[col].fillna(temp_dataset[col].mean())

  # Impute each column with missing values
  all_columns_with_missing = binary_columns + continuous_columns + \
      multi_categorical_columns + categorical_columns

  for col in all_columns_with_missing:
    if dataset[col].isnull().sum() > 0:
      missing_count = dataset[col].isnull().sum()
      print(
          f"Imputing {missing_count} missing values in '{col}' using KNN (k={k})")

      for idx in dataset[dataset[col].isnull()].index:
        # Get current record (with missing value)
        current_record = temp_dataset.loc[idx].values

        # Calculate distances to all complete records in the target column
        distances = []
        complete_records = dataset[dataset[col].notnull()]

        for other_idx in complete_records.index:
          if other_idx == idx:
            continue

          other_record = temp_dataset.loc[other_idx].values

          # Calculate Euclidean distance
          distance = NP.sqrt(
              NP.sum((current_record - other_record) ** 2))
          distances.append((distance, other_idx))

        # Sort by distance and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest_indices = [idx for dist, idx in distances[:k]]

        # Impute with mean (for continuous) or mode (for categorical/binary)
        if col in continuous_columns:
          impute_value = dataset.loc[k_nearest_indices, col].mean()
        else:  # binary, multi-categorical, or string categorical
          impute_value = dataset.loc[k_nearest_indices, col].mode()
          impute_value = impute_value[0] if len(impute_value) > 0 else (
              0 if col in binary_columns else 'Unknown')

        dataset_imputed.loc[idx, col] = impute_value

  missing_after = dataset_imputed.isnull().sum().sum()
  print(f"\nMissing values before KNN imputation: {missing_before}")
  print(f"Missing values after KNN imputation: {missing_after}")

  return dataset_imputed
