import os
import shutil

# Définir les chemins des dossiers source et destination
source_dir = 'D:/Inria/Datasets/gy6or6_dataset/gy6or6_FULL'
target_dir = "D:/Inria/Datasets/gy6or6_dataset/gy6or6_Audios"

# Vérifier si le dossier de destination existe, sinon le créer
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Parcourir tous les fichiers dans le dossier source
for filename in os.listdir(source_dir):
    # Vérifier si le fichier est un fichier .wav
    if filename.endswith('.wav'):
        # Construire le chemin complet du fichier source
        src_file = os.path.join(source_dir, filename)
        # Construire le chemin complet du fichier de destination
        dst_file = os.path.join(target_dir, filename)
        # Copier le fichier
        shutil.copy(src_file, dst_file)
        print(f"Fichier {filename} copié de {source_dir} à {target_dir}")

print("Tous les fichiers .wav ont été copiés.")

