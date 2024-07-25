import os
import pandas as pd

# Chemin du dossier contenant les fichiers CSV
folder_path = 'chemin/vers/le/dossier'

all_files = os.listdir(folder_path)

csv_files = [file for file in all_files if file.endswith('.csv')]

for csv_file in csv_files:

    file_path = os.path.join(folder_path, csv_file)

    df = pd.read_csv(file_path)

    df_transformed = df.rename(columns={'start': 'onset_s', 'end': 'offset_s'})
    df_transformed = df_transformed[['onset_s', 'offset_s', 'label']]

    audio_filename = df['wave'].iloc[0]

    new_csv_filename = f"{audio_filename}.csv"

    new_file_path = os.path.join(folder_path, new_csv_filename)

    df_transformed.to_csv(new_file_path, index=False)
