import pandas as pd
import glob
import os
from Levenshtein import distance as levenshtein_distance

def extract_labels_from_csv_files(directory):
    """
    Extracts labels from CSV files in a given directory.

    Args:
        directory (str): The directory containing CSV files.

    Returns:
        list: A list of labels extracted from the CSV files.
    """
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    csv_files.sort()
    all_labels = []

    for file in csv_files:
        df = pd.read_csv(file)
        if 'syll' in df.columns:
            labels = df['syll']
            all_labels.extend(map(str, labels))  # Convert labels to strings
        else:
            print(f"La colonne 'label' n'existe pas dans le fichier {file}")

    return all_labels

def extract_labels_from_reference_csv(file_path):
    """
    Extracts labels from a reference CSV file.

    Args:
        file_path (str): The path to the reference CSV file.

    Returns:
        list: A list of labels extracted from the reference CSV file.

    Raises:
        ValueError: If the 'label' column does not exist in the reference file.
    """
    df = pd.read_csv(file_path)
    if 'syll' in df.columns:
        labels = df['syll']
        return list(map(str, labels))  # Convert labels to strings
    else:
        raise ValueError("La colonne 'label' n'existe pas dans le fichier de référence")

def compare_labels(directory_path, reference_file_path):
    """
    Compares labels extracted from a directory of CSV files and a reference CSV file.

    Args:
        directory_path (str): The directory containing CSV files.
        reference_file_path (str): The path to the reference CSV file.

    Returns:
        tuple: A tuple containing the Levenshtein distance, syllable error rate, and similarity ratio.
    """
    # Extract labels from the directory and reference file
    labels_list_from_directory = extract_labels_from_csv_files(directory_path)
    labels_list_from_reference = extract_labels_from_reference_csv(reference_file_path)

    # Compute Levenshtein distance between the two sequences
    distance = levenshtein_distance(labels_list_from_directory, labels_list_from_reference)

    # Calculate syllable error rate
    syllable_error_rate = (distance / len(labels_list_from_reference)) * 100

    # Calculate similarity ratio
    max_length = max(len(labels_list_from_directory), len(labels_list_from_reference))
    similarity_ratio = 1 - (distance / max_length)

    return distance, syllable_error_rate, similarity_ratio

# Specify the paths to the directory and reference file
directory_path = 'PATH'
reference_file_path = 'PATH'

# Perform comparison and get results
distance, syllable_error_rate, similarity_ratio = compare_labels(directory_path, reference_file_path)

# Display the results
print(f"La distance d'édition entre les deux chaînes de caractères est : {distance}")
print(f"Le taux d'erreur des syllabes entre les deux chaînes de caractères est de : {syllable_error_rate:.2f}%")
print(f"Le ratio de similarité entre les deux chaînes de caractères est de : {similarity_ratio:.2f}")
