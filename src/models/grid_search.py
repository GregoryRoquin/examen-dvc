import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from joblib import dump


def main():
    try:
        X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv", header=0)
        y_train = pd.read_csv("data/processed_data/y_train.csv", header=0).values.ravel()

        model = RandomForestRegressor(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],          # Nombre d'arbres dans la forêt
            'max_depth': [None, 10, 20],             # Profondeur maximale des arbres
            'min_samples_split': [2, 5],             # Nombre minimal d'échantillons pour diviser un nœud
            'min_samples_leaf': [1, 2],              # Nombre minimal d'échantillons dans une feuille
            'max_features': ['sqrt', 'log2']         # Nombre de features à considérer pour chaque division
        }

        clf_grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',  # Métrique pour évaluer les performances (R²)
            n_jobs=-1,      # Utilise tous les cœurs disponibles pour accélérer
            verbose=2       # Affiche les détails du processus
        )

        print("Début de l'entraînement avec GridSearchCV...")
        clf_grid.fit(X_train_scaled, y_train)

        print(f"Meilleurs paramètres : {clf_grid.best_params_}")
        print(f"Meilleur score (R²) : {clf_grid.best_score_:.4f}")        

        dump(clf_grid.best_params_, "models/best_params.pkl")

        print("Meilleurs paramètres sauvegardés avec succès !")

    except FileNotFoundError as e:
        print(f"Erreur : Fichier introuvable - {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")   


if __name__ == '__main__':
    main()    