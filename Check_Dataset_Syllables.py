import os
import pandas as pd

# Chemin du dossier contenant les fichiers CSV
dossier_csv = 'D:/Inria/Datasets/gr41rd51_dataset/gr41rd51_FULL_marron1'  # Remplacez par le chemin de votre dossier

# Ensemble pour stocker les syllabes uniques
syllabes_uniques = set()

# Parcours de tous les fichiers dans le dossier
for fichier in os.listdir(dossier_csv):
    if fichier.endswith(".csv"):  # Vérification pour ne traiter que les fichiers CSV
        chemin_fichier = os.path.join(dossier_csv, fichier)
        
        # Lecture du fichier CSV dans un DataFrame
        df = pd.read_csv(chemin_fichier)
        
        # Conversion de la colonne 'syll' en chaînes de caractères et ajout à l'ensemble
        syllabes_uniques.update(df['syll'].astype(str).unique())

# Affichage des syllabes uniques
print("Liste des syllabes uniques trouvées :")
A = sorted(syllabes_uniques)  # Trie directement l'ensemble
print(A)
