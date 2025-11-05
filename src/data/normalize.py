import numpy as np
import pandas as pd
from sklearn import preprocessing

def main():
    try:        
        X_train = pd.read_csv("data/processed_data/X_train.csv", header=0)
        X_test = pd.read_csv("data/processed_data/X_test.csv", header=0)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/processed_data/X_train_scaled.csv", index=False)
        pd.DataFrame(X_test_scaled, columns=X_train.columns).to_csv("data/processed_data/X_test_scaled.csv", index=False)

        print("Standardisation terminée et fichiers sauvegardés avec succès !")
    except FileNotFoundError as e:
        print(f"Erreur : Fichier introuvable - {e}")
    except Exception as e:
        print(f"Erreur inattendue : {e}")   

if __name__ == '__main__':
    main()    


