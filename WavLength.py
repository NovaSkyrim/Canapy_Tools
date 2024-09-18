import os
import wave
import contextlib

def calculer_duree_totale_wav(dossier_source):
    duree_totale = 0.0
    for fichier in os.listdir(dossier_source):
        if fichier.endswith('.wav'):
            chemin_fichier = os.path.join(dossier_source, fichier)
            with contextlib.closing(wave.open(chemin_fichier, 'r')) as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duree_fichier = frames / float(rate)
                duree_totale += duree_fichier
                print(f'{fichier}: {duree_fichier:.2f} secondes')
    return duree_totale

dossier_source = '/home/utilisateur/Documents/Canapy/Experiments/Temp_folder/Temp_Dataset'
duree_totale = calculer_duree_totale_wav(dossier_source)
print(f'Durée totale des fichiers .wav: {duree_totale:.2f} secondes')
