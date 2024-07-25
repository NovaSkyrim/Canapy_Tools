import scipy.io

import numpy as np

# Spécifiez le chemin vers le fichier .mat
mat_file_path = 'D:\Vak-Canary-Dataset\llb11\llb11_annotation.mat'

# Charger le fichier .mat
mat_contents = scipy.io.loadmat(mat_file_path)

# Afficher les clés du dictionnaire
print("Clés du fichier .mat :")
print(mat_contents.keys())

# Explorer chaque clé et sa structure
for key in mat_contents:
    if not key.startswith('__'):  # Ignorer les méta-données de scipy comme __header__, __version__, etc.
        print(f"\nClé: {key}")
        value = mat_contents[key]

        # Afficher le type et la taille de la donnée
        print(f"Type: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"Taille: {value.shape}")

        # Afficher les premières données si c'est un tableau numpy
        if isinstance(value, np.ndarray):
            print(f"Données:\n{value}")

        # Si c'est un tableau de structure, afficher ses champs
        if isinstance(value, np.ndarray) and value.dtype.names:
            print("Champs de la structure:")
            for field in value.dtype.names:
                print(f" - {field}: {value[field]}")
                print(f"   Type: {type(value[field])}")
                print(f"   Taille: {value[field].shape}")
