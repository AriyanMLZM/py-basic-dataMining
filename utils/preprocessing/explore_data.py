from .. import draw_line
from .identify_column_types import identify_column_types


def explore_data(dataset):
  draw_line("Exploring Data")

  print("Data Types:")
  print(dataset.dtypes)

  # Identify all column types
  categorical_columns, binary_columns, continuous_columns, multi_categorical_columns = identify_column_types(
      dataset)

  print(f"\nBinary columns: {binary_columns}")
  print(f"Continuous columns: {continuous_columns}")
  print(f"Multi-class categorical columns: {multi_categorical_columns}")

  return categorical_columns, binary_columns, continuous_columns, multi_categorical_columns
