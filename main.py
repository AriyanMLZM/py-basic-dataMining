from utils import load_csv, explore_data

datasetPath = './dataset/dataset.csv'


def main():
  dataset = load_csv(datasetPath)
  if dataset is not None:
    explore_data(dataset)


if __name__ == '__main__':
  main()
