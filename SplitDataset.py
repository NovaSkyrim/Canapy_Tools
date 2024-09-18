import os
import random
import shutil

def copier_paires_fichiers(dossier_source, dossier_destination, pourcentage=0.2):
    # Vérifier si les dossiers existent
    if not os.path.exists(dossier_source):
        print(f"{dossier_source} doesn't exist.")
        return
    if not os.path.exists(dossier_destination):
        os.makedirs(dossier_destination)  # Créer le dossier destination s'il n'existe pas

    # Lister les fichiers .wav dans le dossier source
    fichiers_wav = [f for f in os.listdir(dossier_source) if f.endswith('.wav')]

    if len(fichiers_wav) == 0:
        print("No .wav file found in the source folder.")
        return

    # Calculer le nombre de fichiers à sélectionner (selon le pourcentage donné)
    n_selection = max(1, int(len(fichiers_wav) * pourcentage))  # Au moins 1 fichier

    # Sélectionner aléatoirement 20 % des fichiers .wav
    fichiers_selectionnes = random.sample(fichiers_wav, n_selection)

    # Copier les fichiers .wav et leurs fichiers .csv correspondants
    for fichier_wav in fichiers_selectionnes:
        # Nom du fichier .csv correspondant
        fichier_csv = fichier_wav.replace('.wav', '.csv')

        # Chemin complet des fichiers source
        chemin_wav_source = os.path.join(dossier_source, fichier_wav)
        chemin_csv_source = os.path.join(dossier_source, fichier_csv)

        # Chemin complet des fichiers destination
        chemin_wav_destination = os.path.join(dossier_destination, fichier_wav)
        chemin_csv_destination = os.path.join(dossier_destination, fichier_csv)

        # Copier les fichiers s'ils existent
        if os.path.exists(chemin_csv_source):
            shutil.copy(chemin_wav_source, chemin_wav_destination)
            shutil.copy(chemin_csv_source, chemin_csv_destination)
        else:
            print(f"{fichier_csv} is missing. Ignored.")
    return

copier_paires_fichiers('/home/utilisateur/Documents/Canapy/Datasets/Marron1Full', '/home/utilisateur/Documents/Canapy/Datasets/Test')
