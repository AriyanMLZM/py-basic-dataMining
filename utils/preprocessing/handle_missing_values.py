from .. import draw_line


# fill missing values for each type of columns
def handle_missing_values(dataset, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns):
  draw_line("Handling Missing Values")

  dataset_clean = dataset.copy()
  missing_before = dataset.isnull().sum().sum()

  if missing_before == 0:
    print("No missing values to handle.")
    return dataset_clean

  # Handle binary columns - use mode
  for col in binary_columns:
    if dataset[col].isnull().sum() > 0:
      missing_count = dataset[col].isnull().sum()
      fill_value = dataset[col].mode()[0] if len(
          dataset[col].mode()) > 0 else 0
      dataset_clean[col] = dataset[col].fillna(fill_value)
      print(
          f"Binary column '{col}': Filled {missing_count} missing values with mode: {fill_value}")

  # Handle continuous columns - use median
  for col in continuous_columns:
    if dataset[col].isnull().sum() > 0:
      missing_count = dataset[col].isnull().sum()
      fill_value = dataset[col].median()
      dataset_clean[col] = dataset[col].fillna(fill_value)
      print(
          f"Continuous column '{col}': Filled {missing_count} missing values with median: {fill_value:.2f}")

  # Handle multi-class categorical columns - use mode
  for col in multi_categorical_columns:
    if dataset[col].isnull().sum() > 0:
      missing_count = dataset[col].isnull().sum()
      fill_value = dataset[col].mode()[0] if len(
          dataset[col].mode()) > 0 else 0
      dataset_clean[col] = dataset[col].fillna(fill_value)
      print(
          f"Multi-categorical column '{col}': Filled {missing_count} missing values with mode: {fill_value}")

  # Handle string categorical columns - use mode
  for col in categorical_columns:
    if dataset[col].isnull().sum() > 0:
      missing_count = dataset[col].isnull().sum()
      fill_value = dataset[col].mode()[0] if len(
          dataset[col].mode()) > 0 else 'Unknown'
      dataset_clean[col] = dataset[col].fillna(fill_value)
      print(
          f"String categorical column '{col}': Filled {missing_count} missing values with mode: {fill_value}")

  missing_after = dataset_clean.isnull().sum().sum()
  print(f"\nMissing values before: {missing_before}")
  print(f"Missing values after: {missing_after}")

  return dataset_clean
