from utils import load_csv, identify_column_types, draw_line
from utils.eda import plot_binary_distributions, plot_multi_categorical_distributions, plot_qqplots, plot_histograms, plot_boxplots, plot_correlation_matrix

# Paths
dataset_path = './output/preprocessed_data.csv'
plots_path = './plots/'


def main():
  dataset = load_csv(dataset_path)
  if dataset is not None:
    # Dividing the attributs
    numerical_columns, categorical_columns, binary_columns, continuous_columns, multi_categorical_columns = identify_column_types(
        dataset)

    draw_line(f"Plotting - saved in {plots_path}")
    # Plotting
    plot_qqplots(dataset, continuous_columns, plots_path)
    plot_histograms(dataset, continuous_columns, plots_path)
    plot_boxplots(dataset, continuous_columns, plots_path)
    plot_binary_distributions(dataset, binary_columns, plots_path)
    plot_multi_categorical_distributions(
        dataset, multi_categorical_columns, plots_path)
    plot_correlation_matrix(dataset, numerical_columns, plots_path)

  else:
    print('No preprocessed data was found.\nRun main_preprocessing.py first or just run main.py')


if __name__ == "__main__":
  main()
