from utils import load_csv, explore_data, detect_missing_values, handle_missing_values, detect_outliers, handle_outliers

datasetPath = './dataset/dataset.csv'


def main():
  dataset = load_csv(datasetPath)
  if dataset is not None:
    numerical_columns, categorical_columns, binary_columns, continuous_columns, multi_categorical_columns = explore_data(
        dataset)
    detect_missing_values(dataset)
    dataset_clean = handle_missing_values(
        dataset, binary_columns, continuous_columns, multi_categorical_columns, categorical_columns
    )
    detect_outliers(dataset_clean, continuous_columns)
    handle_outliers(dataset_clean, continuous_columns)


if __name__ == '__main__':
  main()
