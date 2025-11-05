import numpy as np
import pandas as pd
from sklearn import preprocessing
import click
import os

@click.command()
@click.argument('x_train_filepath', type=click.Path(exists=False), required=True)
@click.argument('x_test_filepath', type=click.Path(exists=False), required=True)
@click.argument('scaled_path', type=click.Path(exists=False), required=True)
def main(x_train_filepath, x_test_filepath, scaled_path):
    try:        
        X_train = pd.read_csv(x_train_filepath, header=0)
        X_test = pd.read_csv(x_test_filepath, header=0)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(os.path.join(scaled_path,"X_train_scaled.csv"), index=False)
        pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(os.path.join(scaled_path,"X_test_scaled.csv"), index=False)

        print("Standardisation terminée et fichiers sauvegardés avec succès !")
    except FileNotFoundError as e:
        print(f"Erreur : Fichier introuvable - {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")   

if __name__ == '__main__':
    main()    


