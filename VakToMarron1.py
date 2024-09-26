import os
import pandas as pd

folder_path = 'Chemin du dossier contenant les fichiers .csv'
output_directory = "output/"

for filename in os.listdir(folder_path):
    if filename.endswith('.wav.csv'):
        input_filepath = os.path.join(folder_path, filename)

        df = pd.read_csv(input_filepath)

        df.insert(0, '', range(0, len(df)))

        df.insert(1, 'wave', filename.replace('.wav.csv', '.wav'))

        df.columns = ['', 'wave', 'start', 'end', 'syll']

        output_filepath = os.path.join(output_directory, filename.replace('.wav.csv', '.csv'))

        df.to_csv(output_filepath, index=False)
