import librosa

def analyser_audio(wav_path):
    # Charger le fichier audio
    y, sr = librosa.load(wav_path, sr=None)  # sr=None pour conserver le taux d'échantillonnage d'origine

    # Calculer le hop length (longueur de saut)
    # Le hop length est la distance entre les échantillons consécutifs dans une représentation de type STFT
    # On peut estimer le hop length en utilisant le nombre de frames et la durée du fichier
    # Nous allons calculer ce nombre en utilisant librosa.feature.mfcc
    hop_length = librosa.get_duration(y=y, sr=sr) / (len(librosa.feature.mfcc(y=y, sr=sr).T))

    # Afficher les résultats
    print(f"Taux d'échantillonnage (sampling rate) : {sr} Hz")
    print(f"Longueur de saut (hop length) : {hop_length:.2f} s")

# Exemple d'utilisation
wav_path = '/home/utilisateur/Documents/Tweetynet/Canary_Dataset/llb11_dataset_marron1/llb11_00224_2018_05_04_12_51_23.wav'
analyser_audio(wav_path)

