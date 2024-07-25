import os
import numpy as np
import matplotlib.pyplot as plt

def display_spectrograms_from_folder(folder_path):
    # Lister tous les fichiers .npz dans le dossier
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

    # Parcourir chaque fichier .npz
    for file_name in npz_files:
        file_path = os.path.join(folder_path, file_name)

        # Charger le fichier .npz
        data = np.load(file_path)

        # Vérifier les clés disponibles dans le fichier
        print(f"Fichier: {file_name}, Clés: {data.files}")

        # Assumer que les données MFCC sont sous la clé 'feature'
        if 'feature' in data.files:
            mfcc_data = data['feature']

            # Vérifier la forme des données
            print(f"Forme du tableau MFCC: {mfcc_data.shape}")

            # Afficher le spectrogramme
            plt.figure(figsize=(10, 6))
            plt.imshow(mfcc_data.T, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time Frames')
            plt.ylabel('MFCC Coefficients')
            plt.title(f'MFCC Spectrogram - {file_name}')
            plt.show()
        else:
            print(f"Clé 'feature' non trouvée dans {file_name}")

# Spécifiez le chemin du dossier contenant les fichiers .npz
folder_path = 'C:/Users/clemd/OneDrive/Bureau/CanapyNathan/canapy/bird1_output/spectrograms'  # Remplacez par votre chemin
# Appeler la fonction pour afficher les spectrogrammes
display_spectrograms_from_folder(folder_path)
