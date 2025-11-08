from utils.preprocessing import explore_data, detect_missing_values, handle_missing_values, detect_outliers, handle_outliers, normalize_data, report_processed_quality, knn_handle_missing_values
from utils import load_csv, save_csv

# Paths
datasetPath = './dataset/dataset.csv'
outputPath = './output/processed_data.csv'
# Configs
use_knn_missing_values = True  # Boolean
outlier_method = 'cap'  # cap, remove, ignore
normalize_method = 'minmax'  # minmax, zscore


def main():
  # Load dataset
  dataset = load_csv(datasetPath)

  if dataset is not None:
    # Divide attributes
    categorical_columns, binary_columns, continuous_columns, multi_categorical_columns = explore_data(
        dataset)

    # Handle missing values
    detect_missing_values(dataset)
    if use_knn_missing_values:
      dataset_imputed = knn_handle_missing_values(
          dataset, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns)
    else:
      dataset_imputed = handle_missing_values(
          dataset, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns
      )

    # Handling outliers
    detect_outliers(dataset_imputed, continuous_columns)
    dataset_noOutliers = handle_outliers(
        dataset_imputed, continuous_columns, normalize_method)

    # Normalize
    dataset_normalized = normalize_data(
        dataset_noOutliers, continuous_columns, normalize_method
    )

    # Report output quality
    report_processed_quality(
        dataset_normalized, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns
    )

    # Save output
    save_csv(dataset_normalized, outputPath)


if __name__ == '__main__':
  main()
