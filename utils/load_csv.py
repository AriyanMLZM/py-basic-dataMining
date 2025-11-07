import pandas as PD


def load_csv(datasetPath):
  print("=" * 30)
  print("Loading Dataset")
  print("=" * 30)
  try:
    dataset = PD.read_csv(datasetPath)
    print(f"\n{dataset.shape[0]} rows and {dataset.shape[1]} columns loaded\n")
    return dataset
  except:
    print("Error Loading Dataset")
    return None
