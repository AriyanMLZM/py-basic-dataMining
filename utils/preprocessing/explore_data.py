from .. import draw_line, identify_column_types


def explore_data(dataset):
  draw_line("Exploring Data")

  # Basic info
  print("Dataset Info:")
  print(f"Shape: {dataset.shape}")

  print("\nFirst 5 rows:")
  print(dataset.head())

  print("\nData Types:")
  print(dataset.dtypes)

  # Identify all column types
  numerical_columns, categorical_columns, binary_columns, continuous_columns, multi_categorical_columns = identify_column_types(
      dataset)

  print(f"\nNumerical columns: {numerical_columns}")
  print(f"String categorical columns: {categorical_columns}\n")
  print(f"Binary columns: {binary_columns}")
  print(f"Continuous columns: {continuous_columns}")
  print(f"Multi-class categorical columns: {multi_categorical_columns}")

  # Dataset description for continuous columns only
  if continuous_columns:
    print("\nContinuous Columns Description:")
    print(dataset[continuous_columns].describe())

  # Binary columns distribution
  if binary_columns:
    print("\nBinary Columns Distribution:")
    for col in binary_columns:
      value_counts = dataset[col].value_counts().sort_index()
      percentages = (dataset[col].value_counts(
          normalize=True) * 100).sort_index()
      print(f"{col}: {dict(value_counts)} - {dict(percentages.round(2))}%")

  # Multi-categorical columns distribution
  if multi_categorical_columns:
    print("\nMulti-class Categorical Columns Distribution:")
    for col in multi_categorical_columns:
      value_counts = dataset[col].value_counts().sort_index()
      percentages = (dataset[col].value_counts(
          normalize=True) * 100).sort_index()
      print(f"{col}: {dict(value_counts)} - {dict(percentages.round(2))}%")

  return categorical_columns, binary_columns, continuous_columns, multi_categorical_columns
