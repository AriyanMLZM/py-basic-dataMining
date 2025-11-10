import pandas as PD
from .. import draw_line


def one_hot_encode_categoricals(dataset, multi_categorical_columns, categorical_columns):
  draw_line("One-Hot-Encoding Categoricals")

  df_encoded = dataset.copy()
  encoded_columns = []

  # Combine all categorical columns
  all_categorical_columns = multi_categorical_columns + categorical_columns

  # Store target column separately
  target_data = df_encoded['target'].copy()
  # Remove target from dataframe temporarily
  df_encoded = df_encoded.drop(columns=['target'])

  if not all_categorical_columns:
    print("No categorical columns to encode")
    return df_encoded, encoded_columns

  for col in all_categorical_columns:
    if col in df_encoded.columns:
      # Get unique values and convert to integers to remove decimals
      unique_vals = df_encoded[col].unique()
      # Convert to integers if they're floats
      unique_vals_clean = [int(val) if isinstance(
          val, float) else val for val in unique_vals]
      print(
          f"Encoding '{col}' with {len(unique_vals)} categories: {sorted(unique_vals_clean)}")

      # Create clean column names without decimals
      dummies = PD.get_dummies(
          df_encoded[col], prefix=col, prefix_sep='_')

      # Clean the column names to remove .0 from floats
      clean_column_names = {}
      for dummy_col in dummies.columns:
        if '.' in dummy_col:
          # Remove .0 from column names like "ca_0.0" -> "ca_0"
          clean_name = dummy_col.replace('.0', '')
          clean_column_names[dummy_col] = clean_name
        else:
          clean_column_names[dummy_col] = dummy_col

      # Rename the columns
      dummies = dummies.rename(columns=clean_column_names)
      dummies = dummies.astype(int)  # Convert to 1/0

      # Add to dataframe
      df_encoded = PD.concat([df_encoded, dummies], axis=1)

      # Drop original column
      df_encoded = df_encoded.drop(columns=[col])

      # Track encoded columns
      encoded_columns.extend(dummies.columns.tolist())

  # Add target again to end
  df_encoded['target'] = target_data

  print(f"\nOriginal shape: {dataset.shape}")
  print(f"After encoding: {df_encoded.shape}")

  return df_encoded, encoded_columns
