# Import des librairies nécessaires
import pandas as pd  
import numpy as np  
import glob          
import os         
import shutil     
import sys         

# Ajout du chemin vers canapy
sys.path.append('chemin/vers/canapy')

# Importation des modules
from canapy import Corpus
from canapy.annotator import SynAnnotator, Annotator
from Levenshtein import distance as levenshtein_distance

# Fonction pour créer un DataFrame à partir d'annotations CSV
def create_frame_annotations(csv_path, sampling_rate, hop_length):
    """Crée un DataFrame contenant les annotations d'un fichier CSV, et nettoie les colonnes."""
    df = pd.read_csv(csv_path)  # Lecture du fichier CSV dans un DataFrame
    
    # Nettoyer les colonnes pour supprimer les espaces
    df.columns = df.columns.str.strip()

    # Vérification de la présence des colonnes 'start' et 'end'
    if 'start' not in df.columns or 'end' not in df.columns:
        print(f"Les colonnes 'start' ou 'end' sont manquantes dans le fichier : {csv_path}")
        return pd.DataFrame()  # Retourne un DataFrame vide si elles sont absentes
    
    return df  # Retourne le DataFrame avec les annotations


# Fonction pour étendre les annotations au format "frame" et comparer deux DataFrames
def as_frame_comparison(gold_df, pred_df, hop_length, sampling_rate):
    """Étendre gold_df au format frame pour comparaison avec pred_df."""
    
    # Nettoyage des colonnes
    gold_df.columns = gold_df.columns.str.strip()
    pred_df.columns = pred_df.columns.str.strip()
    
    # Calcul des frames de début et de fin à partir des annotations
    try:
        gold_df["start_frame"] = (gold_df["start"] * sampling_rate / hop_length).astype(int)
        gold_df["end_frame"] = (gold_df["end"] * sampling_rate / hop_length).astype(int)
        pred_df["start_frame"] = (pred_df["start"] * sampling_rate / hop_length).astype(int)
        pred_df["end_frame"] = (pred_df["end"] * sampling_rate / hop_length).astype(int)
    except KeyError as e:
        print(f"Erreur lors de la comparaison des frames : {e}")
        print("La comparaison n'a pas pu être effectuée.")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur

    # Création des labels pour chaque frame
    gold_frames = []
    pred_frames = []

    # Boucle sur chaque fichier audio
    for wav in gold_df['wave'].unique():
        print(f"Traitement du fichier audio : {wav}")
        
        # Séparation des DataFrames en fonction du fichier audio
        gold_wav_df = gold_df[gold_df['wave'] == wav]
        pred_wav_df = pred_df[pred_df['wave'] == wav]
        
        # Calcul du nombre total de frames
        max_frames = int(np.ceil(gold_wav_df['end'].max() * sampling_rate / hop_length))
        
        # Initialisation des labels avec "SIL" (silence)
        gold_labels = np.full(max_frames, "SIL", dtype=object)
        pred_labels = np.full(max_frames, "SIL", dtype=object)
        
        # Remplissage des frames avec les labels corrects pour gold et pred
        for _, row in gold_wav_df.iterrows():
            gold_labels[row['start_frame']:row['end_frame']] = row['syll']
        for _, row in pred_wav_df.iterrows():
            pred_labels[row['start_frame']:row['end_frame']] = row['syll']
        
        # Création de DataFrames pour chaque frame
        gold_frame_df = pd.DataFrame({'frame': np.arange(max_frames), 'label': gold_labels, 'wave': wav})
        pred_frame_df = pd.DataFrame({'frame': np.arange(max_frames), 'label': pred_labels, 'wave': wav})

        # Ajout aux listes de frames
        gold_frames.append(gold_frame_df)
        pred_frames.append(pred_frame_df)
    
    # Concatenation des résultats pour tous les fichiers audio
    gold_frames = pd.concat(gold_frames)
    pred_frames = pd.concat(pred_frames)

    # Fusion des DataFrames pour comparaison
    comparison_df = pd.merge(gold_frames, pred_frames, on=['frame', 'wave'], suffixes=('_gold', '_pred'))

    return comparison_df


# Fonction pour calculer le Frame Error Rate (FER)
def calculate_frame_error_rate(gold_annotations_path, pred_annotations_path):
    """Calcule le Frame Error Rate entre les annotations gold et prédictions."""
    
    # Charger les annotations gold et pred
    gold_annotations = create_frame_annotations(gold_annotations_path, sampling_rate=44100, hop_length=512)
    pred_annotations = create_frame_annotations(pred_annotations_path, sampling_rate=44100, hop_length=512)

    # Comparer les annotations au format frame
    comparison_df = as_frame_comparison(gold_annotations, pred_annotations, hop_length=512, sampling_rate=44100)

    # Calcul du Frame Error Rate (FER)
    total_frames = len(comparison_df)
    incorrect_frames = np.sum(comparison_df['label_gold'] != comparison_df['label_pred'])
    
    return incorrect_frames / total_frames if total_frames > 0 else float('nan')


# Fonction pour extraire les labels des fichiers CSV d'un répertoire
def extract_labels_from_csv_files(directory):
    """Extrait les labels des fichiers CSV dans un répertoire."""
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    csv_files.sort()  
    all_labels = []

    # Parcourir les fichiers CSV et extraire les labels
    for file in csv_files:
        df = pd.read_csv(file)
        if 'syll' in df.columns:
            labels = df['syll']
        elif 'label' in df.columns:
            labels = df['label']
        else:
            print(f"La colonne 'syll' ou 'label' n'existe pas dans le fichier {file}")
            continue

        # Filtrer les labels inutiles (SIL, TRASH)
        filtered_labels = labels[(labels != 'SIL') & (labels != 'TRASH')]
        all_labels.extend(map(str, filtered_labels))

    return all_labels


# Fonction pour comparer les labels extraits d'un répertoire par rapport à un fichier de référence
def compare_labels(directory_path, reference_file_path):
    """Compare les labels d'un répertoire avec un fichier de référence en calculant la distance de Levenshtein."""
    
    # Extraction des labels des fichiers CSV
    labels_list_from_directory = remove_consecutive_duplicates(extract_labels_from_csv_files(directory_path))
    labels_list_from_reference = remove_consecutive_duplicates(extract_labels_from_csv_files(reference_file_path))
    
    # Calcul de la distance de Levenshtein entre les deux listes de labels
    distance = levenshtein_distance(labels_list_from_directory, labels_list_from_reference)

    # Calcul du Syllable Error Rate (SER)
    syllable_error_rate = (distance / len(labels_list_from_reference)) * 100

    return syllable_error_rate


# Fonction pour supprimer les doublons consécutifs dans une liste
def remove_consecutive_duplicates(s):
    """Supprime les doublons consécutifs dans une liste."""
    result = s[0] if s else ""  # Initialise le résultat avec le premier élément s'il existe
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:  # Ajoute l'élément à result s'il est différent du précédent
            result += s[i]
    return result


def optim_hp(config_canapy, config_modifiee, dataset_path):    
    try:
        # Remplacement du fichier de configuration existant par un nouveau
        if os.path.exists(config_canapy):
            os.remove(config_canapy)
            shutil.copy(config_modifiee, config_canapy)

    except FileNotFoundError as e:
        print(f"Erreur: {e}")
    except Exception as e:
        print(f"Erreur inattendue: {e}")

    output_path = "annotator"

    # Création du corpus à partir du répertoire de données
    corpus = Corpus.from_directory(
        audio_directory=dataset_path,
        annots_directory=dataset_path,
        annot_format="marron1csv",
        audio_ext=".wav",
    )

    # Entraînement du modèle
    annotator = SynAnnotator()
    annotator.fit(corpus)

    # Sauvegarde du modèle
    annotator.to_disk(output_path)

    # Suppression des spectrogrammes existants, s'ils existent (sinon Canapy bug, Nathan est au courant)
    spectrograms_path = os.path.join(dataset_path, "spectrograms")
    if os.path.exists(spectrograms_path):
        shutil.rmtree(spectrograms_path)

    # Chargement du modèle et prédiction des annotations
    annotator = Annotator.from_disk(output_path)
    corpus_avec_annotations = annotator.predict(corpus)

    # Sauvegarde des annotations
    temp_output_path = "annot_output/"
    corpus_avec_annotations.to_directory(temp_output_path)
    print("Annotations terminées et enregistrées dans", temp_output_path)

    # Comparaison des labels et calcul des erreurs
    syllable_error_rate = compare_labels(temp_output_path, dataset_path)
    
    # Calcul du frame_error_rate pour chaque fichier CSV
    frame_error_rates = []
    for csv_file in os.listdir(temp_output_path):
        csv_file1 = os.path.join(dataset_path, csv_file)
        csv_file2 = os.path.join(temp_output_path, csv_file)
        frame_error_rate = calculate_frame_error_rate(csv_file1, csv_file2)
        frame_error_rates.append(frame_error_rate)

    # Stats globales des frame_error-rate
    frame_error_rate_mean = np.mean(frame_error_rates)
    frame_error_rate_median = np.median(frame_error_rates)

    print(syllable_error_rate, frame_error_rate_mean, frame_error_rate_median)

    return syllable_error_rate, frame_error_rate_mean, frame_error_rate_median

if __name__ == '__main__':

    # Définition des chemins pour les données et les fichiers de configuration
    dataset_path = "D:/Inria/Datasets/M1-2016-spring" # Qui contient les fichiers .csv et les audios .wav
    config_modifiee = "D:/Inria/canapy/config/store/modified.config.yml"
    config_canapy = "D:/Inria/canapy/config/store/default.config.yml"

    optim_hp(config_canapy, config_modifiee, dataset_path)
