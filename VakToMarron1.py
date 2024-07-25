import os
import pandas as pd

# Chemin du dossier contenant les fichiers .csv
folder_path = 'C:/Users/clemd/OneDrive/Bureau/Pro/INRIA/INRIA/Recherche/ML_ComparisonTests/Vak_Dataset_Test'

for filename in os.listdir(folder_path):
    if filename.endswith('.wav.csv'):
        input_filepath = os.path.join(folder_path, filename)

        df = pd.read_csv(input_filepath)

        df.insert(0, '', range(0, len(df)))

        df.insert(1, 'wave', filename.replace('.wav.csv', '.wav'))

        df.columns = ['', 'wave', 'start', 'end', 'syll']

        output_filepath = os.path.join('C:/Users/clemd/OneDrive/Bureau/Pro/INRIA/INRIA/Recherche/ML_ComparisonTests/output', filename.replace('.wav.csv', '.csv'))

        df.to_csv(output_filepath, index=False)
