import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import load, dump

def main():
    try:
        X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv", header=0)
        y_train = pd.read_csv("data/processed_data/y_train.csv", header=0).values.ravel()

        best_params = load("models/best_params.pkl")

        model = RandomForestRegressor(**best_params, random_state=42)

        model.fit(X_train_scaled, y_train)

        dump(model, "models/gbr_model.pkl")

        print("Modèle sauvegardé avec succès !")

    except FileNotFoundError as e:
        print(f"Erreur : Fichier introuvable - {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")

if __name__ == '__main__':
    main()