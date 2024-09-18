import os
import pandas as pd
from pydub import AudioSegment

# Chemin du dossier contenant les fichiers CSV et WAV
dossier = '/home/utilisateur/Documents/Canapy/Marron1Set'

# Obtenez les listes des fichiers CSV et WAV, et triez-les par nom
fichiers_csv = [f for f in os.listdir(dossier) if f.endswith('.csv')]
fichiers_wav = [f for f in os.listdir(dossier) if f.endswith('.wav')]

fichiers_csv.sort()
fichiers_wav.sort()

# Initialiser une liste pour stocker les DataFrames
dataframes = []

# Initialiser le temps de décalage à 0
offset_time = 0

# Initialiser un objet AudioSegment vide pour fusionner les fichiers WAV
audio_fusionne = AudioSegment.silent(duration=0)

# Parcourir chaque fichier CSV et WAV correspondants
for fichier_csv, fichier_wav in zip(fichiers_csv, fichiers_wav):
    # Charger le CSV dans un DataFrame
    df = pd.read_csv(os.path.join(dossier, fichier_csv))
    
    # Ajouter l'offset_time aux colonnes start et end
    df['start'] += offset_time
    df['end'] += offset_time
    
    # Mettre à jour offset_time pour le prochain fichier
    offset_time = df['end'].iloc[-1]
    
    # Ajouter le DataFrame ajusté à la liste
    dataframes.append(df)
    
    # Charger le fichier WAV correspondant
    audio_segment = AudioSegment.from_wav(os.path.join(dossier, fichier_wav))
    
    # Ajouter l'audio segmenté à l'audio fusionné
    audio_fusionne += audio_segment

# Concaténer tous les DataFrames
df_concatene = pd.concat(dataframes, ignore_index=True)

# Sauvegarder le CSV fusionné et ajusté
df_concatene.to_csv(os.path.join(dossier, 'fichier_fusionne.csv'), index=False)

# Exporter le fichier audio fusionné en WAV
audio_fusionne.export(os.path.join(dossier, 'fichier_fusionne.wav'), format='wav')

print("Fusion et ajustement des fichiers CSV et WAV terminés.")

