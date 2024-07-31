import pandas as pd
import glob
import os
from Levenshtein import distance as levenshtein_distance


def extract_labels_from_csv_files(directory):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    csv_files.sort()
    all_labels = []

    for file in csv_files:
        df = pd.read_csv(file)
        if 'label' in df.columns:
            labels = df['label']
            all_labels.extend(map(str, labels))
        else:
            print(f"La colonne 'label' n'existe pas dans le fichier {file}")

    return all_labels


def extract_labels_from_reference_csv(file_path):
    df = pd.read_csv(file_path)
    if 'label' in df.columns:
        labels = df['label']
        return list(map(str, labels))
    else:
        raise ValueError("La colonne 'label' n'existe pas dans le fichier de référence")


def compare_labels(directory_path, reference_file_path):
    labels_list_from_directory = extract_labels_from_csv_files(directory_path)
    labels_list_from_reference = extract_labels_from_reference_csv(reference_file_path)

    distance = levenshtein_distance(labels_list_from_directory, labels_list_from_reference)

    syllable_error_rate = (distance / len(labels_list_from_reference)) * 100

    return distance, syllable_error_rate


directory_path = 'PATH'
reference_file_path = 'PATH'

distance, syllable_error_rate = compare_labels(directory_path, reference_file_path)
print(f"La distance d'édition entre les deux chaînes de caractères est : {distance}")
print(f"Le taux d'erreur des syllabes entre les deux chaînes de caractères est de : {syllable_error_rate}%")
