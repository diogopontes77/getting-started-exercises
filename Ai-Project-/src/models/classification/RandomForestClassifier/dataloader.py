import pandas as pd

def load_data(path="DatasetCredit-g.csv"):
    mapping_foreign = {'no': 0, 'yes': 1}
    mapping = {'bad': 0, 'good': 1}

    df = pd.read_csv(path)
    df['class_binary'] = df['class'].map(mapping)
    df['foreign'] = df['foreign_worker'].map(mapping_foreign)
    df = df.drop(columns=['foreign_worker', 'class'])

    X = df.drop(columns=["class_binary"])
    y = df["class_binary"]
    return X, y