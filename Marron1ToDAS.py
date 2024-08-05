import os
from pydub import AudioSegment
import pandas as pd


def merge_wav_files(directory, output_file):
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    if not wav_files:
        print("Aucun fichier .wav trouvé dans le répertoire.")
        return None

    wav_files.sort()

    combined = AudioSegment.silent(duration=0)  # Initialiser avec une piste silencieuse

    for wav_file in wav_files:
        audio = AudioSegment.from_wav(os.path.join(directory, wav_file))

        # Chercher le fichier CSV correspondant
        csv_file = wav_file.replace('.wav', '.csv')
        csv_path = os.path.join(directory, csv_file)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['start'] = pd.to_numeric(df['start'], errors='coerce')
            df['end'] = pd.to_numeric(df['end'], errors='coerce')

            # Vérification de validité des annotations
            invalid_rows = df[(df['start'].isna()) | (df['end'].isna()) | (df['start'] >= df['end'])]
            if not invalid_rows.empty:
                print(f"Lignes invalides dans {csv_file}:")
                print(invalid_rows)

            # Supprimer les lignes non valides
            df.dropna(subset=['start', 'end'], inplace=True)
            df = df[df['start'] < df['end']]

            # Nouveau audio combiné pour le fichier en cours
            file_combined = AudioSegment.silent(duration=0)

            last_end = 0

            for _, row in df.iterrows():
                start = row['start'] * 1000  # Convertir en millisecondes
                end = row['end'] * 1000

                if start < end and start < len(audio):
                    # Assurez-vous que l'intervalle est valide
                    if end > len(audio):
                        print(
                            f"Fin de l'intervalle ajustée dans {csv_file}, ligne {_ + 1} : start={start}, end={end}, len(audio)={len(audio)}")
                        end = len(audio)

                    # Ajouter la partie valide de l'audio
                    file_combined += audio[start:end]

            combined += file_combined

        else:
            print(f"Fichier CSV correspondant non trouvé pour {wav_file}. Audio ignoré.")

    combined.export(output_file, format='wav')
    print(f"Les fichiers WAV ont été fusionnés et sauvegardés sous {output_file}")

    return combined.duration_seconds / 1000  # Retourner la durée en secondes


def merge_and_transform_csv_files(directory, output_file, total_duration):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    if not csv_files:
        print("Aucun fichier .csv trouvé dans le répertoire.")
        return

    csv_files.sort()

    dataframes = []
    time_offset = 0.0

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(directory, csv_file))

        df['start'] = pd.to_numeric(df['start'], errors='coerce')
        df['end'] = pd.to_numeric(df['end'], errors='coerce')

        # Vérification de validité des annotations
        invalid_rows = df[(df['start'].isna()) | (df['end'].isna()) | (df['start'] >= df['end'])]
        if not invalid_rows.empty:
            print(f"Lignes invalides dans {csv_file}:")
            print(invalid_rows)

        # Supprimer les lignes non valides
        df.dropna(subset=['start', 'end'], inplace=True)
        df = df[df['start'] < df['end']]

        if not df.empty:
            if time_offset > 0:
                # Calculer le début et la fin du silence
                silence_start = time_offset
                silence_end = df['start'].iloc[0]
                if silence_start < silence_end:
                    silence_df = pd.DataFrame({
                        'name': ['SIL'],
                        'start_seconds': [silence_start],
                        'stop_seconds': [silence_end],
                        'channel': [0]
                    })
                    dataframes.append(silence_df)

            df['start'] += time_offset
            df['end'] += time_offset

            time_offset = df['end'].iloc[-1]

            dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    if merged_df['end'].iloc[-1] > total_duration:
        print("Attention : La durée totale des fichiers CSV dépasse la durée du fichier WAV fusionné.")
        print(f"Durée CSV: {merged_df['end'].iloc[-1]} > Durée Audio: {total_duration}")

    merged_df = merged_df.rename(columns={'syll': 'name', 'start': 'start_seconds', 'end': 'stop_seconds'})
    merged_df['channel'] = 0
    merged_df = merged_df[['name', 'start_seconds', 'stop_seconds', 'channel']]

    merged_df.to_csv(output_file, index=False)
    print(f"Les fichiers CSV ont été fusionnés, transformés et sauvegardés sous {output_file}")

    return merged_df


directory = 'Dataset Marron1 PATH'
output_wav_file = 'combined.wav'
output_csv_file = 'merged.csv'

total_duration = merge_wav_files(directory, output_wav_file)

merged_df = merge_and_transform_csv_files(directory, output_csv_file, total_duration)
print(merged_df.tail())
