import os
import shutil
import subprocess
import toml
import glob

# Définition des chemins
BASE_PATH = "/home/utilisateur/Documents/VakStudy/Canary_Dataset/Training30"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
PREP_TRAIN_PATH = os.path.join(BASE_PATH, "prep/train")
PREP_OUTPUT_PATH = os.path.join(BASE_PATH, "prep_output")
VAK_FINAL_OUTPUT_PATH = os.path.join(BASE_PATH, "vak_final_output")
NOTES_PATH = os.path.join(BASE_PATH, "Notes")
TOML_TRAIN_PATH = os.path.join(BASE_PATH, "gy6or6_train.toml")
TOML_PREDICT_PATH = os.path.join(BASE_PATH, "gy6or6_predict.toml")

# Configuration des durées d'entraînement
train_dur_values = [240, 300, 360, 420, 480]

# Fonction pour nettoyer un répertoire
def clear_directory(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Erreur lors de la suppression de {file_path}. Raison: {e}")

# Fonction pour nettoyer le fichier de notes
def clear_notes(path):
    if os.path.exists(path):
        with open(path, "w") as f:
            f.truncate(0)  # Effacer le contenu du fichier

# Fonction pour modifier le fichier .toml
def modify_toml_train(toml_path, train_dur):
    config = toml.load(toml_path)
    config['vak']['prep']['train_dur'] = train_dur
   # if 'path' in config['vak']['train']['dataset']:
   #     del config['vak']['train']['dataset']['path']
    with open(toml_path, "w") as toml_file:
        toml.dump(config, toml_file)

def modify_toml_predict(toml_path, new_checkpoint_path):
    config = toml.load(toml_path)
    config['vak']['predict']['checkpoint_path'] = new_checkpoint_path
    config['vak']['predict']['labelmap_path'] = new_checkpoint_path.replace("checkpoints", "labelmap")
    config['vak']['predict']['frames_standardizer_path'] = new_checkpoint_path.replace("checkpoints", "frames_standardizer")
    if 'path' in config['vak']['predict']['dataset']:
        del config['vak']['predict']['dataset']['path']
    with open(toml_path, "w") as toml_file:
        toml.dump(config, toml_file)

# Fonction pour exécuter une commande bash et capturer la sortie
def run_command(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Erreur lors de l'exécution de la commande: {command}\n{result.stderr}")
    return result.stdout

# Fonction principale pour le processus d'entraînement
def run_training_process(train_dur, index):
    # Étapes 1 : Nettoyer les répertoires
    print("Nettoyage des repertoires")
    print("Nettoyage de prep train")
    clear_directory(PREP_TRAIN_PATH)
    print("Nettoyage de prep output")
    clear_directory(PREP_OUTPUT_PATH)
    print("Nettoyage de train path")
    clear_directory(TRAIN_PATH)
    print("Nettoyage de final output")
    clear_directory(VAK_FINAL_OUTPUT_PATH)

    # Étape 2 : Nettoyer le fichier Notes
    print("Nettoyage de la note")
    clear_notes(NOTES_PATH)

    # Étape 3 : Modifier le fichier .toml pour l'entraînement
    print("Modification du train.toml")
    modify_toml_train(TOML_TRAIN_PATH, train_dur)

    # Étape 4 : Exécuter les commandes d'entraînement
    print("Preparation du dataset d'entrainement")
    os.chdir(BASE_PATH)
    print("vak prep gy6or6_train.toml")
    run_command("vak prep gy6or6_train.toml")
    run_command("clear")
    print("vak train gy6or6_train.toml")
    run_command("vak train gy6or6_train.toml")

    # Étape 5 : Copier la sortie de la console dans Notes
    training_output = run_command("vak train gy6or6_train.toml")
    with open(NOTES_PATH, "a") as notes_file:
        notes_file.write(training_output)

    # Étape 6 : Récupérer le nom du dossier créé
    results_dirs = glob.glob(os.path.join(TRAIN_PATH, "results_*"))
    if not results_dirs:
        print("Erreur : aucun dossier de résultats trouvé.")
        return
    latest_result_dir = max(results_dirs, key=os.path.getmtime)
    latest_result_name = os.path.basename(latest_result_dir)

    # Étape 7 : Modifier le fichier .toml pour la prédiction
    new_checkpoint_path = os.path.join(TRAIN_PATH, latest_result_name, "TweetyNet/checkpoints/max-val-acc-checkpoint.pt")
    modify_toml_predict(TOML_PREDICT_PATH, new_checkpoint_path)

    # Étape 8 : Exécuter les commandes de prédiction
    print("vak prep gy6or6_predict.toml")
    run_command("vak prep gy6or6_predict.toml")
    print("vak predict gy6or6_predict.toml")
    run_command("vak predict gy6or6_predict.toml")

    # Étape 9 : Calculer la distance d'édition et enregistrer dans Notes
    os.chdir("/home/utilisateur/Documents/VakStudy/Canary_Dataset")
    run_command("clear")
    edit_distance_output = run_command("python EditDistanceCalculator_Vak.py")
    with open(NOTES_PATH, "a") as notes_file:
        notes_file.write(edit_distance_output)

    # Étape 10 : Dupliquer et renommer le dossier Training30
    new_training_dir = f"Training30_{index}_{train_dur}"
    shutil.copytree(BASE_PATH, os.path.join(BASE_PATH, "..", new_training_dir))

# Boucle pour entraîner le modèle pour chaque valeur de train_dur
for index, train_dur in enumerate(train_dur_values, start=1):
    run_training_process(train_dur, index)