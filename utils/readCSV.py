import pandas as PD


def readCSV():
  try:
    dataset = PD.read_csv("./dataset/dataset.csv")
    print('Success Reading Dataset \nFirst Row:')
    print(dataset.head(1))
    return dataset
  except:
    print("Error Reading Dataset")
    return None
