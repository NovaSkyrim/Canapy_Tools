import os
import pandas as pd

folder_path = 'D:/Inria/Datasets/gy6or6_dataset/gy6or6_FULL'
output_folder_path = 'D:/Inria/Datasets/gy6or6_dataset/gy6or6_marron1'

os.makedirs(output_folder_path, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith('.wav.csv'):
        input_filepath = os.path.join(folder_path, filename)

        # Lire le fichier CSV
        df = pd.read_csv(input_filepath)

        # Ajouter la colonne d'index
        df.insert(0, '', range(0, len(df)))

        # Ajouter la colonne 'wave'
        df.insert(1, 'wave', filename.replace('.wav.csv', '.wav'))

        # Renommer les colonnes
        df.columns = ['', 'wave', 'start', 'end', 'syll']

        # Définir le chemin de sortie
        output_filepath = os.path.join(output_folder_path, filename.replace('.wav.csv', '.csv'))

        # Sauvegarder le DataFrame modifié dans le dossier de sortie
        df.to_csv(output_filepath, index=False)

