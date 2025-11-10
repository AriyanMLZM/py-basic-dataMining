from utils.preprocessing import explore_data, detect_missing_values, handle_missing_values, detect_outliers, handle_outliers, normalize_data, knn_handle_missing_values, one_hot_encode_categoricals
from utils.eda import plot_binary_distributions, plot_qqplots, plot_histograms, plot_boxplots, plot_correlation_matrix, plot_multi_categorical_distributions
from utils import load_csv, save_csv, draw_line

# Paths
dataset_path = './dataset/dataset.csv'
output_path = './output/preprocessed_data.csv'
plots_path = './plots/'
# Options
use_knn_missing_values = True  # Boolean
outlier_method = 'cap'  # cap, remove, ignore
normalize_method = 'minmax'  # minmax, zscore


def main():
  # Load dataset
  dataset = load_csv(dataset_path)

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

    plot_boxplots(dataset_imputed, continuous_columns, plots_path + 'before_')
    # Handling outliers
    detect_outliers(dataset_imputed, continuous_columns)
    dataset_noOutliers = handle_outliers(
        dataset_imputed, continuous_columns, outlier_method)

    # Plotting
    draw_line(f"Plotting - saved in {plots_path}")
    plot_boxplots(dataset_noOutliers, continuous_columns,
                  plots_path + 'after_')
    plot_qqplots(dataset_noOutliers, continuous_columns, plots_path)
    plot_histograms(dataset_noOutliers, continuous_columns, plots_path)
    plot_multi_categorical_distributions(
        dataset_noOutliers, multi_categorical_columns, plots_path)
    plot_binary_distributions(dataset_noOutliers, binary_columns, plots_path)

    # Normalize
    dataset_normalized = normalize_data(
        dataset_noOutliers, continuous_columns, normalize_method
    )

    # One-Hot-Encoding for categoricals
    dataset_encoded, encoded_columns = one_hot_encode_categoricals(
        dataset_normalized, multi_categorical_columns, categorical_columns
    )
    binary_columns = binary_columns + encoded_columns

    # Correlation Matrix
    plot_correlation_matrix(dataset_encoded, continuous_columns +
                            binary_columns, target_column='target', save_path=plots_path)

    # Save output
    save_csv(dataset_encoded, output_path)


if __name__ == '__main__':
  main()
