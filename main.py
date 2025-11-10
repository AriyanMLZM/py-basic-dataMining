from main_preprocessing import main as preprocessing
from main_classification import main as classification


def main():
  print("\n# Preprocessing")
  preprocessing()
  print("\n# Classification")
  classification()


if __name__ == '__main__':
  main()
