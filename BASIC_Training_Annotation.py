import csv
import sys
import os
import shutil
import time
import re
import pandas as pd

sys.path.append('D:/Inria/canapy')  # Add the path of the canapy folder
from canapy import Corpus
from canapy.annotator import SynAnnotator
from canapy.annotator import Annotator

def modify_seed(file_path, new_seed):  # Modify the model seed
    try:
        # Read the current content of the file
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # Update the content
        updated_content = re.sub(r'(init_esn_model\([^,]+,[^,]+,[^,]+,)\s*\d+', r'\1 ' + str(new_seed), file_content)
        
        # Write the updated content to a temporary file
        temp_file_path = file_path + '.tmp'
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(updated_content)
        
        # Replace the old file with the temporary file
        os.replace(temp_file_path, file_path)
        
        # Verify the update by reading the file again
        with open(file_path, 'r') as file:
            final_content = file.read()
        
        # Check if the update has been correctly applied
        if re.search(r'(init_esn_model\([^,]+,[^,]+,[^,]+,)\s*' + str(new_seed), final_content):
            print(f"Seed successfully changed to {new_seed} in file {file_path}")
        else:
            print(f"Failed to update the seed in file {file_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def copy_corrections(split_csv, working_directory, bird_name, seed):  # Copy the original annotations corresponding to the test and train sets into new folders
    
    train_correction_folder = f'{working_directory}/{bird_name}/{seed}/Train_correction'
    test_correction_folder = f'{working_directory}/{bird_name}/{seed}/Test_correction'
    os.makedirs(train_correction_folder, exist_ok=True)
    os.makedirs(test_correction_folder, exist_ok=True)

    # Read the CSV file
    with open(split_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Check the value of the 'train' column
            train_value = row['train'].strip().lower() == 'true'

            # Path of the annotated file (annot_path column)
            annot_file = row['annot_path']

            # Determine the destination folder
            dest_folder = train_correction_folder if train_value else test_correction_folder

            # Full path of the file in the destination folder
            dest_file_path = os.path.join(dest_folder, os.path.basename(annot_file))

            # Check if the file already exists in the destination folder
            if not os.path.exists(dest_file_path):
                if os.path.exists(annot_file):
                    # Copy the file if it does not already exist in the destination folder
                    shutil.copy(annot_file, dest_file_path)
                    print(f"File {annot_file} copied to {dest_folder}")
                else:
                    print(f"Source file {annot_file} not found.")

def move_common_csv_files(folder1, folder2, destination_folder, new_folder_name1):  # Separates predicted annotations into train and test sets based on the correction folder
    # Create the destination folder if it does not already exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # List CSV files in both folders
    files_folder1 = {f for f in os.listdir(folder1) if f.endswith('.csv')}
    files_folder2 = {f for f in os.listdir(folder2) if f.endswith('.csv')}
    
    # Find common files
    common_files = files_folder1.intersection(files_folder2)
    
    # Move common files to the destination folder
    for file in common_files:
        file_path = os.path.join(folder1, file)
        shutil.move(file_path, os.path.join(destination_folder, file))
        print(f"{file} moved to {destination_folder}")
    
    # Rename the first folder
    os.rename(folder1, new_folder_name1)
    print(f"Folder {folder1} renamed to {new_folder_name1}")

dataset_path = "/home/user/Documents/Canapy/Datasets/Marron1Full"  # Folder containing the annotations
audio_path = "/home/user/Documents/Canapy/Datasets/Marron1Full"  # Here the folder includes both annotations and audios
annotator_init_path = "D:/Inria/canapy/canapy/annotator/synannotator.py"  # Location of the syntactic model initialization to modify its seed
working_directory = "Path/To/Your/Working/Directory"  # Path to your working directory
bird_name = "Marron1"  # Name of the working folder to be created
seeds_values = list(range(1, 11))  # List of seed values

if __name__ == '__main__':
    for seed in seeds_values:
        modify_seed(annotator_init_path, seed)

        output_path = f"{working_directory}/{bird_name}/{seed}/Annots"
        os.makedirs(output_path, exist_ok=True)

        save_model_path = f"{working_directory}/{bird_name}/{seed}/annotator"

        corpus = Corpus.from_directory(  # Define the corpus
            audio_directory=audio_path,
            annots_directory=dataset_path,
            annot_format="marron1csv",  # Canapy supports all Crowsetta formats
            audio_ext=".wav",  # Canapy supports all Librosa formats
        )

        print(corpus.dataset)  # Display the train/test split dataframe

        save_split_path = f"{working_directory}/{bird_name}/{seed}/Split.csv" 

        corpus.dataset.to_csv(save_split_path)  # Save the train/test split

        annotator = SynAnnotator()  # Define the model; here it is a syntactic model

        annotator.fit(corpus)  # Train the model

        print(annotator.vocab)  # Display the list of labels considered by the model

        annotator.to_disk(save_model_path)  # Save the model

        spectrograms_dir = os.path.join(audio_path, "spectrograms")  # Remove the temporary spectrogram folder to avoid an error in Canapy
        if os.path.exists(spectrograms_dir):
            shutil.rmtree(spectrograms_dir)

        # Part for predicting annotations of unannotated audios

        corpus = Corpus.from_directory(audio_directory=audio_path)  # New corpus with only unannotated audios

        annotator = Annotator.from_disk(save_model_path)  # Load the trained model

        corpus_with_annotations = annotator.predict(corpus)  # Predict annotations

        corpus_with_annotations.to_directory(output_path)  # Save annotations

        print("Annotations completed and saved!")

        copy_corrections(save_split_path, working_directory, bird_name, seed)  # Copy corrections of annotations corresponding to the same split

        train_correction_folder = f'{working_directory}/{bird_name}/{seed}/Train_correction'
        train_set = f'{working_directory}/{bird_name}/{seed}/Train_set'
        test_set = f'{working_directory}/{bird_name}/{seed}/Test_set'
        os.makedirs(train_correction_folder, exist_ok=True)
        os.makedirs(train_set, exist_ok=True)
        os.makedirs(test_set, exist_ok=True)

        move_common_csv_files(output_path, train_correction_folder, train_set, test_set)  # Split annotations into Test_set and Train_set
