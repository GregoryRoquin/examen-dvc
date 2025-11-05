import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import load
import json

def main():
    try:
        model = load("models/gbr_model.pkl")

        X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv", header=0)
        y_test = pd.read_csv("data/processed_data/y_test.csv", header=0).values.ravel()

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

        with open("metrics/scores.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("Métriques sauvegardées dans 'metrics/scores.json' !")

        print("Création du dataset avec prédictions...")
        X_test_with_predictions = X_test_scaled.copy()
        X_test_with_predictions["y_true"] = y_test
        X_test_with_predictions["y_pred"] = y_pred

        X_test_with_predictions.to_csv("data/prediction.csv", index=False)
        print("Dataset avec prédictions sauvegardé dans 'data/prediction.csv' !")

    except FileNotFoundError as e:
        print(f"Erreur : Fichier introuvable - {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")

if __name__ == '__main__':
    main()