import os
import pandas as pd
from Levenshtein import distance


def get_syllable_sequence(file_path):
    """
    Lit un fichier CSV et extrait la séquence de syllabes en tant que chaîne unique,
    en ignorant les silences marqués par 'SIL'.

    Args:
        file_path (str): Chemin du fichier CSV.

    Returns:
        str: Séquence de syllabes concaténées, sans les silences.
    """
    df = pd.read_csv(file_path)
    df = df[df['syll'] != 'SIL']
    syllables = df['syll'].tolist()
    syllable_sequence = ''.join(syllables)
    return syllable_sequence


def calculate_edit_distances(folder1, folder2):
    """
    Calcule les distances d'édition entre les séquences de syllabes des fichiers dans deux dossiers.

    Args:
        folder1 (str): Chemin vers le premier dossier contenant les fichiers CSV annotés.
        folder2 (str): Chemin vers le second dossier contenant les fichiers CSV calculés par l'algorithme.

    Returns:
        dict: Dictionnaire associant les noms de fichiers audio à leurs distances d'édition.
        int: Distance d'édition totale entre tous les fichiers correspondants.
    """
    edit_distances = {}
    total_edit_distance = 0

    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    file_to_syllables = {}

    for file in files1:
        if file.endswith('.csv'):
            file_path = os.path.join(folder1, file)
            syllable_sequence = get_syllable_sequence(file_path)
            audio_file_name = pd.read_csv(file_path)['wave'].iloc[0]
            file_to_syllables[audio_file_name] = syllable_sequence

    for file in files2:
        if file.endswith('.csv'):
            file_path = os.path.join(folder2, file)
            calculated_sequence = get_syllable_sequence(file_path)
            audio_file_name = pd.read_csv(file_path)['wave'].iloc[0]
            if audio_file_name in file_to_syllables:
                annotated_sequence = file_to_syllables[audio_file_name]
                edit_distance = distance(annotated_sequence, calculated_sequence)
                edit_distances[audio_file_name] = edit_distance
                total_edit_distance += edit_distance
            else:
                print(f"Warning: No matching file for {audio_file_name} found in folder1.")

    return edit_distances, total_edit_distance


folder1 = 'Chemin vers le 1er dossier d"annotations'
folder2 = 'Chemin vers le 2ème dossier d"annotations'

edit_distances, total_edit_distance = calculate_edit_distances(folder1, folder2)

for audio_file, dist in edit_distances.items():
    print(f"Audio File: {audio_file}, Edit Distance: {dist}")

print(f"\nTotal Edit Distance: {total_edit_distance}")