import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data(input_path: str, output_path: str, train_fraction: float) -> None:
    """Splits data into features and target training and test sets.

    Args:
        input_path: Path to csv file containing features and target.
        output_path: Path to directory where the training and test sets should be stored
        train_fraction: Fraction of the dataset allocated to the training set.
    """

    # TODO:
    # Step 1: Read data from csv file (using Pandas)
    df = pd.read_csv(input_path)
    
    # Step 2: Split the data into training and test sets. Use argument `train_fraction` to define the size of the training set.
    y = df["Species"]
    X= df.drop(columns=["Species"])
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size= train_fraction, random_state = 42)
    
    # Step 3: Split the training data into features and targets (X_train and Y_train respectively). The target column is `species`.
    # ja está
    
    # Step 4: Repeat Step 3 to the test data.
    
    
    # Step 5: Save X_train, X_test, Y_train, Y_test to output_path as ´X_train.csv´, `X_test.csv`, `Y_train.csv`, and `Y_test.csv` respectively.
    X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "Y_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "Y_test.csv"), index=False)

    
# Do not edit!
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Engineering node.')
    parser.add_argument('input_path', type=str, help='Path to csv file containing features and target')
    parser.add_argument('output_path', type=str, help='Path to directory where the training and test sets should be stored')
    parser.add_argument('train_fraction', type=float, help='Fraction of the dataset allocated to the training set')

    args = parser.parse_args()
    split_data(input_path=args.input_path, output_path=args.output_path, train_fraction=args.train_fraction)
