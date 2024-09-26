import csv
import sys
import os
import shutil
import time
import re
import pandas as pd
sys.path.append('D:/Inria/canapy') # On ajoute le chemin du dossier canapy
from canapy import Corpus
from canapy.annotator import SynAnnotator
from canapy.annotator import Annotator

def modifier_seed(file_path, new_seed):
    try:
        # Lire le contenu actuel du fichier
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # Mettre à jour le contenu
        updated_content = re.sub(r'(init_esn_model\([^,]+,[^,]+,[^,]+,)\s*\d+', r'\1 ' + str(new_seed), file_content)
        
        # Écrire le contenu mis à jour dans un fichier temporaire
        temp_file_path = file_path + '.tmp'
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(updated_content)
        
        # Remplacer l'ancien fichier par le fichier temporaire
        os.replace(temp_file_path, file_path)
        
        # Vérifier la mise à jour en relisant le fichier
        with open(file_path, 'r') as file:
            final_content = file.read()
        
        # Vérifier si la mise à jour a été correctement appliquée
        if re.search(r'(init_esn_model\([^,]+,[^,]+,[^,]+,)\s*' + str(new_seed), final_content):
            print(f"Seed modifiée avec succès à {new_seed} dans le fichier {file_path}")
        else:
            print(f"Échec de la mise à jour de la seed dans le fichier {file_path}")
        
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

def copy_corrections(split_csv, working_directory, bird_name, seed) :
    
    train_correction_folder = f'{working_directory}/{bird_name}/{seed}/Train_correction'
    test_correction_folder = f'{working_directory}/{bird_name}/{seed}/Test_correction'
    os.makedirs(train_correction_folder, exist_ok=True)
    os.makedirs(test_correction_folder, exist_ok=True)

    # Lire le fichier CSV
    with open(split_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Vérifie la valeur de la colonne 'train'
            train_value = row['train'].strip().lower() == 'true'

            # Chemin du fichier annoté (colonne annot_path)
            annot_file = row['annot_path']

            # Déterminer le dossier de destination
            dest_folder = train_correction_folder if train_value else test_correction_folder

            # Chemin complet du fichier dans le dossier de destination
            dest_file_path = os.path.join(dest_folder, os.path.basename(annot_file))

            # Vérifier si le fichier existe déjà dans le dossier de destination
            if not os.path.exists(dest_file_path):
                if os.path.exists(annot_file):
                    # Copier le fichier s'il n'existe pas déjà dans le dossier de destination
                    shutil.copy(annot_file, dest_file_path)
                    print(f"Fichier {annot_file} copié dans {dest_folder}")
                else:
                    print(f"Fichier source {annot_file} non trouvé.")

def deplacer_fichiers_csv_communs(dossier1, dossier2, dossier_destination, nouveau_nom_dossier1):
    # Créer le dossier de destination s'il n'existe pas déjà
    if not os.path.exists(dossier_destination):
        os.makedirs(dossier_destination)
    
    # Lister les fichiers CSV dans les deux dossiers
    fichiers_dossier1 = {f for f in os.listdir(dossier1) if f.endswith('.csv')}
    fichiers_dossier2 = {f for f in os.listdir(dossier2) if f.endswith('.csv')}
    
    # Trouver les fichiers en commun
    fichiers_communs = fichiers_dossier1.intersection(fichiers_dossier2)
    
    # Déplacer les fichiers communs vers le dossier de destination
    for fichier in fichiers_communs:
        chemin_fichier = os.path.join(dossier1, fichier)
        shutil.move(chemin_fichier, os.path.join(dossier_destination, fichier))
        print(f"{fichier} déplacé vers {dossier_destination}")
    
    # Renommer le premier dossier
    os.rename(dossier1, nouveau_nom_dossier1)
    print(f"Dossier {dossier1} renommé en {nouveau_nom_dossier1}")

dataset_path = "/home/utilisateur/Documents/Canapy/Datasets/Marron1Full"
spectrograms_dir = os.path.join(dataset_path, "spectrograms")
annotator_init_path = "D:/Inria/canapy/canapy/annotator/synannotator.py"
working_directory = "Path/To/Your/Working/Directory"
bird_name = "Marron1"
seeds_values = list(range(1, 11))

if __name__ == '__main__':

    for seed in seeds_values:

        modifier_seed(annotator_init_path, seed)

        output_path = f"{working_directory}/{bird_name}/{seed}/Annots"
        save_model_path = f"{working_directory}/{bird_name}/{seed}/annotator"

        corpus = Corpus.from_directory(
            audio_directory=dataset_path,
            annots_directory=dataset_path,
            annot_format="marron1csv",
            audio_ext=".wav",
        )

        print(corpus.dataset)

        save_split_path = f"{working_directory}/{bird_name}/{seed}/Split.csv"

        corpus.dataset.to_csv(save_split_path)

        annotator = SynAnnotator()

        annotator.fit(corpus)

        print(annotator.vocab)

        annotator.to_disk(save_model_path)

        if os.path.exists(spectrograms_dir):
            shutil.rmtree(spectrograms_dir)

        corpus = Corpus.from_directory(audio_directory=dataset_path)

        annotator = Annotator.from_disk(save_model_path)

        corpus_avec_annotations = annotator.predict(corpus)

        corpus_avec_annotations.to_directory(output_path)

        print("Annotations terminées et enregistrées !")

        copy_corrections(save_split_path)

        train_correction_folder = f'{working_directory}/{bird_name}/{seed}/Train_correction'

        train_set = f'{working_directory}/{bird_name}/{seed}/Train_set'
        test_set = f'{working_directory}/{bird_name}/{seed}/Test_set'
        os.makedirs(train_set, exist_ok=True)
        os.makedirs(test_set, exist_ok=True)

        deplacer_fichiers_csv_communs(output_path, train_correction_folder, train_set, test_set)