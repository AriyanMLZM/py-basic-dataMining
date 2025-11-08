from utils.preprocessing import explore_data, detect_missing_values, handle_missing_values, detect_outliers, handle_outliers, normalize_data, report_processed_quality, knn_handle_missing_values
from utils import load_csv, save_csv

datasetPath = './dataset/dataset.csv'


def main():
  dataset = load_csv(datasetPath)
  if dataset is not None:
    categorical_columns, binary_columns, continuous_columns, multi_categorical_columns = explore_data(
        dataset)
    detect_missing_values(dataset)
    # dataset_clean = handle_missing_values(
    #     dataset, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns
    # )
    dataset_imputed = knn_handle_missing_values(
        dataset, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns)
    detect_outliers(dataset_imputed, continuous_columns)
    dataset_noOutliers = handle_outliers(dataset_imputed, continuous_columns)
    dataset_normalized = normalize_data(
        dataset_noOutliers, continuous_columns, method='minmax'
    )
    report_processed_quality(
        dataset_normalized, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns
    )
    save_csv(dataset_normalized, './output/processed_data.csv')


if __name__ == '__main__':
  main()
