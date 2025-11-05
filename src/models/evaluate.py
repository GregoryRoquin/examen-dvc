import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import load
import json
import click
import os

@click.command()
@click.argument('x_test_scaled_filepath', type=click.Path(exists=False), required=True)
@click.argument('y_test_filepath', type=click.Path(exists=False), required=True)
@click.argument('models_filepath', type=click.Path(exists=False), required=True)
@click.argument('metrics_path', type=click.Path(exists=False), required=True)
@click.argument('prediction_path', type=click.Path(exists=False), required=True)
def main(x_test_scaled_filepath, y_test_filepath, models_filepath, metrics_path, prediction_path):
    try:
        model = load(models_filepath)

        X_test_scaled = pd.read_csv(x_test_scaled_filepath, header=0)
        y_test = pd.read_csv(y_test_filepath, header=0).values.ravel()

        print("Prédiction des valeurs de test...")
        y_pred = model.predict(X_test_scaled)

        print("Calcul des métriques...")
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
        print(f"Métriques calculées : {metrics}")

        with open(os.path.join(metrics_path,"scores.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        print("Métriques sauvegardées dans 'metrics/scores.json' !")

        print("Création du dataset avec prédictions...")
        X_test_with_predictions = X_test_scaled.copy()
        X_test_with_predictions["y_true"] = y_test
        X_test_with_predictions["y_pred"] = y_pred

        X_test_with_predictions.to_csv(os.path.join(prediction_path,"prediction.csv"), index=False)
        print("Dataset avec prédictions sauvegardé !")

    except FileNotFoundError as e:
        print(f"Erreur : Fichier introuvable - {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")

if __name__ == '__main__':
    main()