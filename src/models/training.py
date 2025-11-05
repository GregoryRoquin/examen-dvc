import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import load, dump
import click
import os

@click.command()
@click.argument('x_train_scaled_filepath', type=click.Path(exists=False), required=True)
@click.argument('y_train_filepath', type=click.Path(exists=False), required=True)
@click.argument('best_params_filepath', type=click.Path(exists=False), required=True)
@click.argument('models_path', type=click.Path(exists=False), required=True)
def main(x_train_scaled_filepath, y_train_filepath, best_params_filepath, models_path):
    try:
        X_train_scaled = pd.read_csv(x_train_scaled_filepath, header=0)
        y_train = pd.read_csv(y_train_filepath, header=0).values.ravel()

        best_params = load(best_params_filepath)

        model = RandomForestRegressor(**best_params, random_state=42)

        model.fit(X_train_scaled, y_train)

        dump(model, os.path.join(models_path,"gbr_model.pkl"))

        print("Modèle sauvegardé avec succès !")

    except FileNotFoundError as e:
        print(f"Erreur : Fichier introuvable - {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")

if __name__ == '__main__':
    main()