import os
import pandas as pd
import time
from pydub import AudioSegment
from canapy import Corpus
from canapy.annotator import SynAnnotator
from canapy.annotator import Annotator

note_file = '/home/utilisateur/Documents/Canapy/canapy/Canapy_benchmark.txt'
for dataset_path in ['/home/utilisateur/Documents/Canapy/Marron1SetSet']:
    
    for training_set_duration in [120]:
        
        train_dataset = dataset_path
        
        # Entrainement du modèle
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
        with open(note_file, 'a') as f:
            f.write(f"Temps d'entrainement de Canapy : {elapsed_time:.2f} secondes\n")

        print(annotator.vocab)

        annotator.to_disk("save_directory/annotator")
        
        # Chargement du corpus audio non annoté
        audio_directory = '/home/utilisateur/Documents/Canapy/canapy/gy6or6_dataset/032312Audios'
        corpus = Corpus.from_directory(audio_directory=audio_directory)

        # Chargement du modèle entraîné
        chemin_vers_modele = "/home/utilisateur/Documents/Canapy/canapy/save_directory/annotator"
        annotator = Annotator.from_disk(chemin_vers_modele)

        # Annotation du corpus
        start_time = time.time()
        corpus_avec_annotations = annotator.predict(corpus)
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(note_file, 'a') as f:
            f.write(f"Temps d'annotation de Canapy : {elapsed_time:.2f} secondes\n")

        # Enregistrement des annotations
        output_directory = "/home/utilisateur/Documents/Canapy/canapy/Auto_marron1_output"
        corpus_avec_annotations.to_directory(output_directory)

        print("Annotations terminées et enregistrées dans", output_directory)
