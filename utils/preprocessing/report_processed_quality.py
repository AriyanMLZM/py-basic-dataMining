from .. import draw_line


def report_processed_quality(dataset, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns):
  draw_line("Reporting Preprocessed Quality")

  print(f"Dataset Shape: {dataset.shape}")
  print(f"Binary columns: {len(binary_columns)}")
  print(f"Continuous columns: {len(continuous_columns)}")
  print(f"Multi-class categorical columns: {len(multi_categorical_columns)}")
  print(f"String categorical columns: {len(categorical_columns)}")

  # Check for any remaining missing values
  remaining_missing = dataset.isnull().sum().sum()
  print(f"\nRemaining missing values: {remaining_missing}")

  # Binary columns summary
  if binary_columns:
    print("\nBinary Columns Summary:")
    for col in binary_columns:
      value_counts = dataset[col].value_counts().sort_index()
      print(f"  {col}: {dict(value_counts)}")

  # Multi-categorical columns summary
  if multi_categorical_columns:
    print("\nMulti-Categorical Columns Summary:")
    for col in multi_categorical_columns:
      value_counts = dataset[col].value_counts().sort_index()
      print(f"  {col}: {dict(value_counts)}")

  # Continuous columns summary
  if continuous_columns:
    print("\nCONTINUOUS Columns Summary (min/mean/max):")
    for col in continuous_columns:
      print(
          f"  {col}: {dataset[col].min():.2f} / {dataset[col].mean():.2f} / {dataset[col].max():.2f}")
