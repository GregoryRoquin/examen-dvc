import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    file_path = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    
    df = pd.read_csv(file_path, header=0)

    y = df["silica_concentrate"]
    X = df.drop(["silica_concentrate", "date"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train.to_csv("data/processed_data/X_train.csv", index=False)
    X_test.to_csv("data/processed_data/X_test.csv", index=False)
    y_train.to_csv("data/processed_data/y_train.csv", index=False)
    y_test.to_csv("data/processed_data/y_test.csv", index=False)

if __name__ == '__main__':
    main()    





