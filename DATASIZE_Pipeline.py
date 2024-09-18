import os
import glob
import shutil
import random
import subprocess
import sys
import time
import datetime
import pandas as pd
import wave
import contextlib
sys.path.append('/home/utilisateur/Documents/Canapy/canapy')
from pydub import AudioSegment
from canapy import Corpus
from canapy.annotator import SynAnnotator
from canapy.annotator import Annotator
from Levenshtein import distance as levenshtein_distance

def create_unique_save_directory(base_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_path, f"save_{timestamp}")
    
    index = 1
    while os.path.exists(save_dir):
        save_dir = os.path.join(base_path, f"save_{timestamp}_{index}")
        index += 1
    
    os.makedirs(save_dir)
    return save_dir
    
def canapy_train_annotate(train_dataset, audio_directory, output_dir, note_path):
    corpus = Corpus.from_directory(
        audio_directory=train_dataset,
        annots_directory=train_dataset,
        annot_format="marron1csv",
        audio_ext=".wav",
    )
    
    print(corpus.dataset)

    annotator = SynAnnotator()
    
    start_time = time.time()
    annotator.fit(corpus)
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(note_path, 'a') as f:
        f.write(f"Temps d'entrainement de Canapy : {elapsed_time:.2f} secondes\n")

    print(annotator.vocab)
    
    model_path = "/home/utilisateur/Documents/Canapy/Experiments/Temp_folder/Temp_Model/annotator"
    annotator.to_disk(model_path)
    
    corpus = Corpus.from_directory(audio_directory=audio_directory)

    annotator = Annotator.from_disk(model_path)

    start_time = time.time()
    corpus_avec_annotations = annotator.predict(corpus)
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(note_path, 'a') as f:
        f.write(f"Temps d'annotation de Canapy : {elapsed_time:.2f} secondes\n")

    corpus_avec_annotations.to_directory(output_dir)

    print("Annotations terminées et enregistrées dans", output_dir)
    
    return
    
def calculer_duree_totale_wav(dossier_source, note_path):
    duree_totale = 0.0
    for fichier in os.listdir(dossier_source):
        if fichier.endswith('.wav'):
            chemin_fichier = os.path.join(dossier_source, fichier)
            with contextlib.closing(wave.open(chemin_fichier, 'r')) as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duree_fichier = frames / float(rate)
                duree_totale += duree_fichier
                with open(note_path, 'a') as f:
                    f.write(f'{fichier}: {duree_fichier:.2f} secondes\n')
    return duree_totale
    
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
    

    
note_path = '/home/utilisateur/Documents/Canapy/Experiments/DATASIZE/gy60r6_phrase_output/Canapy_benchmark_gy60r6_phrase.txt'
training_full = '/home/utilisateur/Documents/Canapy/Datasets/gy6or6_dataset/032212'
training_set = '/home/utilisateur/Documents/Canapy/Experiments/Temp_folder/Temp_Dataset'
test_set = '/home/utilisateur/Documents/Canapy/Datasets/gy6or6_dataset/032312'
audio_directory='/home/utilisateur/Documents/Canapy/Datasets/gy6or6_dataset/032312Audios'
output_dir = '/home/utilisateur/Documents/Canapy/Experiments/Temp_folder/Temp_Output'
save_model = '/home/utilisateur/Documents/Canapy/Experiments/Temp_folder/Temp_Model'
csv_save_path = '/home/utilisateur/Documents/Canapy/Experiments/DATASIZE/gy60r6_phrase_output'
spectrograms_dir1 = '/home/utilisateur/Documents/Canapy/Experiments/Temp_folder/Temp_Dataset/spectrograms'
spectrograms_dir2 = '/home/utilisateur/Documents/Canapy/Datasets/gy6or6_dataset/032312Audios/spectrograms'

source_files = [f for f in os.listdir(training_full) if f.endswith('.wav')]
paired_files_copied = set(os.path.splitext(f)[0] for f in os.listdir(training_set))

pairs_to_copy = 1

while len(paired_files_copied) < len(source_files):
    copied_count = 0
    while copied_count < pairs_to_copy: 
        chosen_file = random.choice(source_files)
        base_name = os.path.splitext(chosen_file)[0]
        if base_name not in paired_files_copied:
            wav_file = os.path.join(training_full, base_name + '.wav')
            csv_file = os.path.join(training_full, base_name + '.csv')
            shutil.copy(wav_file, training_set)
            shutil.copy(csv_file, training_set)
            paired_files_copied.add(base_name)
            copied_count += 1
    
    save_dir_csv = create_unique_save_directory(csv_save_path)
    print(f"Création du dossier de sauvegarde : {save_dir_csv}")
    for item in os.listdir(output_dir):
        src = os.path.join(output_dir, item)
        dst = os.path.join(save_dir_csv, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(f"Contenu de {output_dir} sauvegardé dans {save_dir_csv}.")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    if os.path.exists(save_model):
        shutil.rmtree(save_model)
        os.makedirs(save_model)

    if os.path.exists(spectrograms_dir1):
        shutil.rmtree(spectrograms_dir1)

    if os.path.exists(spectrograms_dir2):
        shutil.rmtree(spectrograms_dir2)

    start_time = time.time()
    canapy_train_annotate(training_set, audio_directory, output_dir, note_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(note_path, 'a') as f:
        f.write(f"Temps de l'étape Canapy : {elapsed_time:.2f} secondes\n")
        
    duree_totale = calculer_duree_totale_wav(training_set, note_path)
    with open(note_path, 'a') as f:
        f.write(f'Durée totale des fichiers .wav: {duree_totale:.2f} secondes\n')
        
    distance, syllable_error_rate = compare_labels(output_dir, test_set)
    
    with open(note_path, 'a') as f:
        f.write(f"La distance d'édition entre les deux chaînes de caractères est : {distance}\n")
        f.write(f"Le taux d'erreur des syllabes entre les deux chaînes de caractères est de : {round(syllable_error_rate, 2)}%\n")
        f.write("\nENTRAINEMENT SUIVANT\n")     
        
