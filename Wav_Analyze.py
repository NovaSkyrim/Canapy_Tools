import contextlib
import os
import wave
import numpy as np

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
    print(f'Durée totale des fichiers .wav: {duree_totale:.2f} secondes')
    return duree_totale

def get_audio_duration(wav_file):
    """Retourne la durée du fichier audio en secondes."""
    with wave.open(wav_file, 'r') as wav_obj:
        frames = wav_obj.getnframes()
        rate = wav_obj.getframerate()
        duration = frames / float(rate)
    return duration

def calculate_audio_stats(directory):
    """Calcule et affiche les statistiques des durées des fichiers audio .wav dans un dossier donné."""
    durations = []

    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            duration = get_audio_duration(file_path)
            durations.append(duration)
    
    if durations:
        durations = np.array(durations)
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        std_duration = np.std(durations)
        
        print(f"Nombre de fichiers: {len(durations)}")
        print(f"Durée moyenne: {mean_duration:.2f} secondes")
        print(f"Médiane: {median_duration:.2f} secondes")
        print(f"Durée minimale: {min_duration:.2f} secondes")
        print(f"Durée maximale: {max_duration:.2f} secondes")
        print(f"Écart-type: {std_duration:.2f} secondes")
    else:
        print("Aucun fichier .wav trouvé dans le dossier spécifié.")

dossier_audio = '/home/utilisateur/Documents/Canapy/Experiments/Temp_folder/Temp_Dataset'
calculate_audio_stats(dossier_audio)
calculer_duree_totale_wav(dossier_audio)
