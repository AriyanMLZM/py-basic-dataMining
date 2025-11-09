from main_preprocessing import main as preprocessing
from main_classification import main as classification
from main_eda import main as eda


def main():
  print("\n# Stage 1: Preprocessing")
  preprocessing()
  print("\n# Stage 2: EDA")
  eda()
  print("\n# Stage 3: Classification")
  classification()


if __name__ == '__main__':
  main()
