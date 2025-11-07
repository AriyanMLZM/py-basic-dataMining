import numpy as NP


# Identify the categorical, Binary and continuous columns
def identify_column_types(dataset):
  numerical_columns = dataset.select_dtypes(
      include=[NP.number]).columns.tolist()
  categorical_columns = dataset.select_dtypes(
      include=['object']).columns.tolist()

  binary_columns = []
  continuous_columns = []
  multi_categorical_columns = []

  # Check numerical columns
  for col in numerical_columns:
    unique_values = dataset[col].dropna().unique()
    if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
      binary_columns.append(col)
    else:
      continuous_columns.append(col)

  # Check for multi-class categorical columns (numerical but with limited categories)
  for col in numerical_columns:
    if col not in binary_columns and col not in continuous_columns:
      continue

    unique_values = dataset[col].dropna().unique()
    # if a numerical column has few unique values it might be categorical
    if len(unique_values) <= 10 and col not in binary_columns:
      if set(unique_values) == set(range(len(unique_values))) or len(unique_values) < 8:
        multi_categorical_columns.append(col)
        if col in continuous_columns:
          continuous_columns.remove(col)

  # Check string categorical
  for col in categorical_columns:
    unique_values = dataset[col].dropna().unique()
    if len(unique_values) > 2:
      multi_categorical_columns.append(col)

  return numerical_columns, categorical_columns, binary_columns, continuous_columns, multi_categorical_columns
