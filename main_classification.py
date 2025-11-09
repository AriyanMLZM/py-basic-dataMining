from utils import load_csv
from utils.knn_classification import split_train_test, find_optimal_k, evaluate_model, knn_predict, plot_k_vs_accuracy

# Paths
preprocessed_path = './output/preprocessed_data.csv'
plot_save_path = './plots/knn_k_accuracy.png'
# Options
max_k = 15


def main():
  # Load preprocessed data
  dataset = load_csv(preprocessed_path)
  if dataset is not None:
    # Split the dataset
    X_train, X_test, y_train, y_test = split_train_test(
        dataset, target_column='target', test_size=0.2)

    # KNN and optimal K
    best_k, k_values, accuracies = find_optimal_k(
        X_train, X_test, y_train, y_test, max_k)
    y_pred = knn_predict(X_train, y_train, X_test, k=best_k)

    # Plot k-acc
    plot_k_vs_accuracy(k_values, accuracies, plot_save_path)

    # Evalute model f1, prec and ...
    evaluate_model(y_test, y_pred)

  else:
    print('No preprocessed data was found.\nRun main_preprocessing.py first or just run main.py')


if __name__ == '__main__':
  main()
