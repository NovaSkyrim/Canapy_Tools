import csv
import re
import sys
sys.path.append('/home/utilisateur/Documents/Canapy/canapy')
import os
import shutil
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob
from canapy import Corpus
from canapy.annotator import SynAnnotator
from canapy.annotator import Annotator

# Defining the base path for all relative directories
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

# Updating sys.path to ensure the Canapy package is included
sys.path.append(os.path.join(BASE_PATH, 'canapy'))

# Function to modify the seed in the syntactic model
def modify_model_seed(new_seed):
    syn_model_path = os.path.join(BASE_PATH, "canapy", "annotator", "synannotator.py")
    try:
        # Reading the content of the file
        with open(syn_model_path, 'r') as file:
            file_content = file.read()

        # Updating the content by replacing the seed value
        updated_content = re.sub(r'(init_esn_model\([^,]+,[^,]+,[^,]+,)\s*\d+', r'\1 ' + str(new_seed), file_content)

        # Writing the updated content to a temporary file
        temp_file_path = syn_model_path + '.tmp'
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(updated_content)

        # Replacing the old file with the new one
        os.replace(temp_file_path, syn_model_path)

        # Verifying if the seed was correctly updated
        with open(syn_model_path, 'r') as file:
            final_content = file.read()

        if re.search(r'(init_esn_model\([^,]+,[^,]+,[^,]+,)\s*' + str(new_seed), final_content):
            print(f"Seed successfully updated to {new_seed} in {syn_model_path}")
        else:
            print(f"Failed to update the seed in {syn_model_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Function to copy datasets for correction based on the 'train' field in the CSV
def copy_correction_datasets(seed):
    # Define paths relative to the base path
    csv_file_path = os.path.join(BASE_PATH, f"Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/Split.csv")
    train_true_folder = os.path.join(BASE_PATH, f"Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/Train_correction")
    train_false_folder = os.path.join(BASE_PATH, f"Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/Test_correction")

    # Ensure the directories exist
    os.makedirs(train_true_folder, exist_ok=True)
    os.makedirs(train_false_folder, exist_ok=True)

    # Reading the CSV file and copying the annotated files based on 'train' value
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            train_value = row['train'].strip().lower() == 'true'
            annot_file = row['annot_path']
            dest_folder = train_true_folder if train_value else train_false_folder
            dest_file_path = os.path.join(dest_folder, os.path.basename(annot_file))

            if not os.path.exists(dest_file_path):
                if os.path.exists(annot_file):
                    shutil.copy(annot_file, dest_file_path)
                    print(f"File {annot_file} copied to {dest_folder}")
                else:
                    print(f"Source file {annot_file} not found.")

# Function to move common CSV files from two directories to a destination directory
def move_common_csv_files(dir1, dir2, dest_dir, new_dir1_name):
    os.makedirs(dest_dir, exist_ok=True)
    files_dir1 = {f for f in os.listdir(dir1) if f.endswith('.csv')}
    files_dir2 = {f for f in os.listdir(dir2) if f.endswith('.csv')}
    common_files = files_dir1.intersection(files_dir2)

    for file in common_files:
        shutil.move(os.path.join(dir1, file), os.path.join(dest_dir, file))
        print(f"{file} moved to {dest_dir}")

    os.rename(dir1, new_dir1_name)
    print(f"Directory {dir1} renamed to {new_dir1_name}")

# Functions for frame annotations and error rate calculations
def create_frame_annotations(csv_path, sampling_rate, hop_length):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if 'start' not in df.columns or 'end' not in df.columns:
        print(f"Missing 'start' or 'end' columns in {csv_path}")
        return pd.DataFrame()

    return df

def as_frame_comparison(gold_df, pred_df, hop_length, sampling_rate):
    gold_df.columns = gold_df.columns.str.strip()
    pred_df.columns = pred_df.columns.str.strip()

    try:
        gold_df["start_frame"] = (gold_df["start"] * sampling_rate / hop_length).astype(int)
        gold_df["end_frame"] = (gold_df["end"] * sampling_rate / hop_length).astype(int)
        pred_df["start_frame"] = (pred_df["start"] * sampling_rate / hop_length).astype(int)
        pred_df["end_frame"] = (pred_df["end"] * sampling_rate / hop_length).astype(int)
    except KeyError as e:
        print(f"Error while comparing frames: {e}")
        return pd.DataFrame()

    gold_frames, pred_frames = [], []

    for wav in gold_df['wave'].unique():
        gold_wav_df = gold_df[gold_df['wave'] == wav]
        pred_wav_df = pred_df[pred_df['wave'] == wav]

        max_frames = int(np.ceil(gold_wav_df['end'].max() * sampling_rate / hop_length))

        gold_labels = np.full(max_frames, "SIL", dtype=object)
        pred_labels = np.full(max_frames, "SIL", dtype=object)

        for _, row in gold_wav_df.iterrows():
            gold_labels[row['start_frame']:row['end_frame']] = row['syll']

        for _, row in pred_wav_df.iterrows():
            pred_labels[row['start_frame']:row['end_frame']] = row['syll']

        gold_frame_df = pd.DataFrame({'frame': np.arange(max_frames), 'label': gold_labels, 'wave': wav})
        pred_frame_df = pd.DataFrame({'frame': np.arange(max_frames), 'label': pred_labels, 'wave': wav})

        gold_frames.append(gold_frame_df)
        pred_frames.append(pred_frame_df)

    comparison_df = pd.merge(pd.concat(gold_frames), pd.concat(pred_frames), on=['frame', 'wave'], suffixes=('_gold', '_pred'))

    return comparison_df[comparison_df['label_gold'] != comparison_df['label_pred']]

def calculate_frame_error_rate(comparison_df):
    total_frames = len(comparison_df)
    incorrect_frames = np.sum(comparison_df['label_gold'] != comparison_df['label_pred'])
    return incorrect_frames / total_frames if total_frames > 0 else float('nan')

# Compute frame error rate statistics
def compute_frame_error_rate_statistics(gold_folder, pred_folder, sampling_rate, hop_length):
    gold_files = sorted([f for f in os.listdir(gold_folder) if f.endswith('.csv')])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.csv')])

    if set(gold_files) != set(pred_files):
        raise ValueError("Files don't match between the two folders.")

    frame_error_rates = []

    for file in gold_files:
        gold_df = create_frame_annotations(os.path.join(gold_folder, file), sampling_rate, hop_length)
        pred_df = create_frame_annotations(os.path.join(pred_folder, file), sampling_rate, hop_length)

        if not gold_df.empty and not pred_df.empty:
            comparison_df = as_frame_comparison(gold_df, pred_df, hop_length, sampling_rate)
            if not comparison_df.empty:
                frame_error_rate = calculate_frame_error_rate(comparison_df)
                frame_error_rates.append(frame_error_rate)

    if frame_error_rates:
        avg_frame_error_rate = np.mean(frame_error_rates)
        std_frame_error_rate = np.std(frame_error_rates)
    else:
        print("Frame Error Rate cannot be calculated.")
        avg_frame_error_rate, std_frame_error_rate = float('nan'), float('nan')

    return avg_frame_error_rate, std_frame_error_rate

# Functions to extract labels from CSV and compute Levenshtein distance
def extract_labels_from_csv_files(directory):
    csv_files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    all_labels = []

    for file in csv_files:
        df = pd.read_csv(file)
        if 'syll' in df.columns:
            labels = df['syll']
        elif 'label' in df.columns:
            labels = df['label']
        else:
            print(f"Missing 'syll' or 'label' column in {file}")
            continue

        filtered_labels = labels[(labels != 'SIL') & (labels != 'TRASH')]
        all_labels.extend(map(str, filtered_labels))

    return all_labels

def remove_consecutive_duplicates(s):
    return ''.join(ch for i, ch in enumerate(s) if i == 0 or s[i] != s[i - 1])

# Main loop to process multiple seeds
for seed in range(1, 11):
    modify_model_seed(seed)

    # Paths for annotations and results
    output_path = os.path.join(BASE_PATH, f"Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/Output_annots")
    model_save_path = os.path.join(BASE_PATH, f"Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/annotator")
    dataset_path = os.path.join(BASE_PATH, "Datasets/Marron1Full")

    # Train the model using Canapy
    corpus = Corpus.from_directory(audio_directory=dataset_path, annots_directory=dataset_path, annot_format="marron1csv", audio_ext=".wav")
    corpus.dataset.to_csv(os.path.join(BASE_PATH, f"Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/Split.csv"))

    annotator = SynAnnotator()
    annotator.fit(corpus)
    annotator.to_disk(model_save_path)

    # Annotate with the trained model
    corpus_to_annotate = Corpus.from_directory(audio_directory=dataset_path)
    annotator = Annotator.from_disk(model_save_path)
    corpus_with_annotations = annotator.predict(corpus_to_annotate)
    corpus_with_annotations.to_directory(output_path)

    # Copy correction datasets and organize files
    copy_correction_datasets(seed)

    # Move common CSV files and organize directories
    train_dir = os.path.join(BASE_PATH, f"Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/Train_correction")
    test_dir = os.path.join(BASE_PATH, f"Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/Test_set")
    move_common_csv_files(output_path, train_dir, test_dir, test_dir)

    print(f"Seed {seed}: Annotations completed and files organized.")

# Metrics calculation and plotting

# Initialisation du DataFrame pour stocker les résultats
results = []

sampling_rate = 44100  # Exemple: taux d'échantillonnage de 44,1 kHz
hop_length = 512       # Exemple: hop length de 512 échantillons

for i in ['Test', 'Train']:

    for seed in range(1,11):

        directory_path = f"/home/utilisateur/Documents/Canapy/Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/{i}_set"
        reference_folder_path = f"/home/utilisateur/Documents/Canapy/Experiments/BASIC/Marron1_BASIC_SPLIT_10iter/{seed}/{i}_correction"

        distance, syllable_error_rate = compare_labels(directory_path, reference_folder_path)
        average_frame_error_rate, std_frame_error_rate = compute_frame_error_rate_statistics(reference_folder_path, directory_path, sampling_rate, hop_length)

        # Ajout des résultats dans la liste
        results.append({
            'Dataset': i,
            'Seed': seed,
            'Syllable Error Rate': syllable_error_rate,
            'Average Frame Error Rate': average_frame_error_rate,
            'STD Frame Error Rate': std_frame_error_rate
        })

        print(f"Métriques de la seed {seed} terminées...")

# Conversion de la liste des résultats en DataFrame
results_df = pd.DataFrame(results)

# Enregistrement dans un fichier CSV
results_df.to_csv('results_summary.csv', index=False)
print(f"Métriques enregistrées sous results_summary.csv")

# Partie création du graphique

# Charger les données du fichier CSV
df = pd.read_csv('results_summary.csv')

# Filtrer les données pour ne garder que celles où 'Dataset' est 'Test'
df_test = df[df['Dataset'] == 'Test']

# Calculer la moyenne et l'écart type de 'Syllable Error Rate' pour chaque valeur unique de 'Sequences' dans le dataset Test
mean_phrase_error_rate_test = df_test.groupby('Sequences')['Syllable Error Rate'].mean()
std_phrase_error_rate_test = df_test.groupby('Sequences')['Syllable Error Rate'].std()

# Générer le graphique
plt.figure(figsize=(10, 6))

# Tracer la moyenne
plt.plot(mean_phrase_error_rate_test.index, mean_phrase_error_rate_test.values, marker='o', label='Moyenne')

# Remplir la zone de l'écart type
plt.fill_between(mean_phrase_error_rate_test.index,
                 mean_phrase_error_rate_test.values - std_phrase_error_rate_test.values,
                 mean_phrase_error_rate_test.values + std_phrase_error_rate_test.values,
                 color='blue', alpha=0.2, label='Écart type')

# Ajouter les labels et un titre
plt.xlabel('Number of Sequences')
plt.ylabel('Average Phrase Error Rate (%)')
plt.title('Phrase Error Rate of Canapy with Marron1')

# Ajuster les ticks de l'axe des abscisses pour correspondre exactement aux séquences
plt.xticks(mean_phrase_error_rate_test.index)  # Utiliser uniquement les séquences présentes

# Afficher la grille
plt.grid(True)

# Ajouter la légende
plt.legend()

# Enregistrer le graphique au format PNG (ou autre format selon les besoins)
plt.savefig('phrase_error_rate_graph.png', format='png', dpi=300)  # DPI = résolution

# Afficher le graphique
plt.show()
