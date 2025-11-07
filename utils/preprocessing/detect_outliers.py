from .. import draw_line


# Detect outliers IQR
def detect_outliers(dataset, continuous_columns):
  draw_line("Detecting Outliers")

  if not continuous_columns:
    print("No continuous columns for outlier detection.")
    return {}

  outlier_info = {}

  for col in continuous_columns:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = dataset[(dataset[col] < lower_bound) |
                       (dataset[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(dataset)) * 100

    outlier_info[col] = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': outlier_count,
        'outlier_percentage': outlier_percentage
    }

    print(f"{col}:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {outlier_count} ({outlier_percentage:.2f}%)")

  return outlier_info
