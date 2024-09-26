import csv
import random
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
    
    train_correction_folder = f'{working_directory}/{bird_name}/{seed}/{sequence}/Train_correction'
    test_correction_folder = f'{working_directory}/{bird_name}/{seed}/{sequence}/Test_correction'
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

def collecter_syllabes_par_fichier(dossier):
    syllabes_par_fichier = {}
    toutes_syllabes = set()

    # Parcourir tous les fichiers CSV dans le dossier
    for fichier in os.listdir(dossier):
        if fichier.endswith('.csv'):
            chemin_fichier = os.path.join(dossier, fichier)
            
            # Lire le fichier CSV
            df = pd.read_csv(chemin_fichier)
            
            # Extraire les syllabes uniques pour chaque fichier
            syllabes_fichier = set(df['syll'].unique())
            syllabes_par_fichier[fichier] = syllabes_fichier
            
            # Ajouter ces syllabes à l'ensemble de toutes les syllabes
            toutes_syllabes.update(syllabes_fichier)

    return syllabes_par_fichier, toutes_syllabes

def trouver_fichiers_minimum_syllabes(syllabes_par_fichier, toutes_syllabes):
    syllabes_couvertes = set()
    fichiers_selectionnes = []

    # Tant que toutes les syllabes ne sont pas couvertes
    while syllabes_couvertes != toutes_syllabes:
        # Sélectionner le fichier qui ajoute le plus de nouvelles syllabes
        meilleur_fichier = None
        nouvelles_syllabes_max = 0

        for fichier, syllabes in syllabes_par_fichier.items():
            # Calculer combien de nouvelles syllabes ce fichier ajoute
            nouvelles_syllabes = syllabes - syllabes_couvertes
            if len(nouvelles_syllabes) > nouvelles_syllabes_max:
                nouvelles_syllabes_max = len(nouvelles_syllabes)
                meilleur_fichier = fichier

        # Ajouter le meilleur fichier à la liste de sélection
        fichiers_selectionnes.append(meilleur_fichier)
        
        # Mettre à jour les syllabes couvertes
        syllabes_couvertes.update(syllabes_par_fichier[meilleur_fichier])
        
        # Supprimer le fichier sélectionné de la liste
        del syllabes_par_fichier[meilleur_fichier]

    return fichiers_selectionnes


def update_train(csv_file, csv_list):
    """
    Met à jour la colonne 'train' dans le fichier CSV pour marquer comme True les lignes
    dont le fichier CSV dans la colonne 'annot_path' apparaît dans la liste csv_list.

    Args:
        csv_file (str): Chemin du fichier CSV d'entrée.
        csv_list (list): Liste des noms de fichiers CSV pour lesquels la colonne 'train' doit être mise à True.
    Returns:
        None
    """
    # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv(csv_file)

    # Extraire seulement le nom du fichier à partir de la colonne 'annot_path'
    df['csv_name'] = df['annot_path'].apply(lambda x: x.split('\\')[-1])

    # Mettre à jour la colonne 'train' selon la présence du nom de fichier dans 'csv_list'
    df['train'] = df['csv_name'].isin(csv_list)

    # Supprimer la colonne temporaire 'csv_name'
    df.drop(columns=['csv_name'], inplace=True)

    # Enregistrer les changements dans un nouveau fichier CSV
    df.to_csv(csv_file, index=False)

    print(f"Mise à jour terminée ! Le fichier a été enregistré sous {csv_file}")

def add_random_files(csv_path, additional_files, sequence,seed, working_directory, bird_name):
    # Lire le fichier CSV
    df = pd.read_csv(csv_path)
    
    # Extraire les fichiers annotés qui ont déjà train = True
    current_true_files = df[df['train'] == True]['annot_path'].unique()
    
    # Extraire les fichiers annotés qui ont train = False (ou qui ne sont pas encore marqués comme True)
    remaining_files = df[~df['annot_path'].isin(current_true_files)]['annot_path'].unique()
    
    # S'assurer que le nombre de fichiers supplémentaires à ajouter n'excède pas le nombre de fichiers restants
    if additional_files > len(remaining_files):
        raise ValueError(f"Le nombre de fichiers supplémentaires ({additional_files}) est supérieur au nombre de fichiers restants disponibles ({len(remaining_files)}).")
    
    # Sélectionner aléatoirement le nombre spécifié de fichiers supplémentaires
    newly_selected_files = random.sample(list(remaining_files), additional_files)
    
    # Mettre à jour la colonne 'train' pour les nouveaux fichiers sélectionnés
    df['train'] = df['annot_path'].apply(lambda x: True if x in newly_selected_files or x in current_true_files else False)
    
    # Sauvegarder le fichier modifié (optionnel : écraser ou créer un nouveau fichier)
    df.to_csv(f"{working_directory}/{bird_name}/{seed}/{sequence}/Split.csv", index=False)
    
    return df

dataset_path = "/home/utilisateur/Documents/Canapy/Datasets/Marron1Full"
spectrograms_dir = os.path.join(dataset_path, "spectrograms")
annotator_init_path = "D:/Inria/canapy/canapy/annotator/synannotator.py"
working_directory = "Path/To/Your/Working/Directory"
bird_name = "Marron1"
seeds_values = list(range(1, 11))
sequences = [7,9,11,13,15,17,19]

if __name__ == '__main__':

    for seed in seeds_values:

        for sequence in sequences :

            modifier_seed(annotator_init_path, seed)

            output_path = f"{working_directory}/{bird_name}/{seed}/{sequence}/Annots"
            save_model_path = f"{working_directory}/{bird_name}/{seed}/{sequence}/annotator"

            corpus = Corpus.from_directory(
                audio_directory=dataset_path,
                annots_directory=dataset_path,
                annot_format="marron1csv",
                audio_ext=".wav",
            )

            print(corpus.dataset)

            save_split_path = f"{working_directory}/{bird_name}/{seed}/{sequence}/Split.csv"

            corpus.dataset.to_csv(save_split_path)

            if sequence == sequences[0] :

                # Étape 1 : Collecter les syllabes présentes dans chaque fichier et toutes les syllabes du dataset
                syllabes_par_fichier, toutes_syllabes = collecter_syllabes_par_fichier(dataset_path)

                # Étape 2 : Trouver le minimum de fichiers nécessaires pour couvrir toutes les syllabes
                fichiers_minimum = trouver_fichiers_minimum_syllabes(syllabes_par_fichier, toutes_syllabes)

                print(f"Le minimum de fichiers nécessaires pour couvrir toutes les syllabes : {fichiers_minimum}")
                print(f"Le nombre de fichiers minimum est de : {len(fichiers_minimum)}")

                update_train(save_split_path, fichiers_minimum)

                quantity_of_files_to_add = sequences[0]-len(fichiers_minimum)

                add_random_files(save_split_path, quantity_of_files_to_add, sequence, seed, working_directory, bird_name)

            else :

                current_index = sequences.index(sequence)
                next_sequence = sequences[current_index + 1]
                quantity_of_files_to_add = next_sequence - sequence

                add_random_files(save_split_path, quantity_of_files_to_add, sequence, seed, working_directory, bird_name)

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

            train_correction_folder = f'{working_directory}/{bird_name}/{seed}/{sequence}/Train_correction'

            train_set = f'{working_directory}/{bird_name}/{seed}/{sequence}/Train_set'
            test_set = f'{working_directory}/{bird_name}/{seed}/{sequence}/Test_set'
            os.makedirs(train_set, exist_ok=True)
            os.makedirs(test_set, exist_ok=True)

            deplacer_fichiers_csv_communs(output_path, train_correction_folder, train_set, test_set)