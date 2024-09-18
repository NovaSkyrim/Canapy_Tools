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
        if 'syll' in df.columns:
            labels = df['syll']
        elif 'label' in df.columns:
            labels = df['label']
        else:
            print(f"La colonne 'syll' ou 'label' n'existe pas dans le fichier {file}")
            continue
            
        filtered_labels = labels[(labels != 'SIL') & (labels != 'TRASH')]
        all_labels.extend(map(str, filtered_labels))

    return all_labels


def compare_labels(directory_path, reference_file_path):
    labels_list_from_directory = remove_consecutive_duplicates(extract_labels_from_csv_files(directory_path))
    labels_list_from_reference = remove_consecutive_duplicates(extract_labels_from_csv_files(reference_file_path))
    
    print(labels_list_from_reference)
    print(labels_list_from_directory)
    
    print(f"Longueur référence = {len(labels_list_from_reference)}")
    print(f"Longueur directory = {len(labels_list_from_directory)}")

    distance = levenshtein_distance(labels_list_from_directory, labels_list_from_reference)

    syllable_error_rate = (distance / len(labels_list_from_reference)) * 100

    return distance, syllable_error_rate


def remove_consecutive_duplicates(s):
    result = s[0] if s else ""
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:
            result += s[i]
    
    return result


directory_path = '/home/utilisateur/Documents/Canapy/canapy/Auto_marron1_output'
reference_folder_path = '/home/utilisateur/Documents/Canapy/canapy/gy6or6_dataset/032312'

distance, syllable_error_rate = compare_labels(directory_path, reference_folder_path)

# Résultats
print(f"La distance d'édition entre les deux chaînes de caractères est : {distance}")
print(f"Le taux d'erreur des syllabes entre les deux chaînes de caractères est de : {round(syllable_error_rate, 2)}%")
print("\nENTRAINEMENT SUIVANT\n")

