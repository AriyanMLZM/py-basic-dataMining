import pandas as PD


def detect_missing_values(dataset):
  print("\n" + "=" * 30)
  print("Detecting Missing Values")
  print("=" * 30)

  missing_values = dataset.isnull().sum()
  missing_percentage = (missing_values / len(dataset)) * 100

  missing_info = PD.DataFrame({
      'Missing Count': missing_values,
      'Missing Percentage': missing_percentage
  })

  print(missing_info[missing_info['Missing Count'] > 0])

  if missing_values.sum() == 0:
    print("No missing values found!")
  else:
    print(f"\nTotal missing values: {missing_values.sum()}")

  return missing_values
