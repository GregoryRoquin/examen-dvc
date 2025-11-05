import pandas as pd
from sklearn.model_selection import train_test_split
import click
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=False)
@click.argument('output_path', type=click.Path(exists=False), required=False)
def main(input_filepath, output_path):
    if not input_filepath:
        input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    if not output_path:
        output_path = click.prompt('Enter the path for the output preprocessed data', type=click.Path())
    
    df = pd.read_csv(input_filepath, header=0)

    y = df["silica_concentrate"]
    X = df.drop(["silica_concentrate", "date"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    os.makedirs(output_path, exist_ok=True)

    X_train.to_csv(os.path.join(output_path,"X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path,"X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path,"y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path,"y_test.csv"), index=False)

if __name__ == '__main__':
    main()    





