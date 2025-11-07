from utils.load_csv import load_csv
from utils.explore_data import explore_data

datasetPath = './dataset/dataset.csv'


def main():
  dataset = load_csv(datasetPath)
  if dataset is not None:
    explore_data(dataset)


if __name__ == '__main__':
  main()
