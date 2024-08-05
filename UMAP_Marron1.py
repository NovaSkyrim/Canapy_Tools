import pandas as pd
import librosa
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import os
import glob

def extract_syllable(y, sr, start, end):
    """Segment audio entre les marqueurs de début et de fin."""
    return y[int(start * sr):int(end * sr)]

def extract_mfcc(syllable, sr, n_mfcc=13):
    """Extraire et normaliser les MFCCs d'une syllabe."""
    n_fft = min(2048, 2**int(np.floor(np.log2(len(syllable)))))  # Ajuste automatiquement n_fft
    hop_length = max(1, len(syllable) // 50)  # Définir un hop_length approprié
    mfccs = librosa.feature.mfcc(y=syllable, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Normaliser la longueur des MFCCs
    if mfccs.shape[1] < 50:
        pad_width = 50 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :50]  # Tronquer à 50

    return mfccs.flatten()

def process_files(audio_path, annotation_path):
    """Traiter un fichier audio et ses annotations pour extraire les caractéristiques."""
    features = []
    syllable_labels = []

    annotations = pd.read_csv(annotation_path)

    y, sr = librosa.load(audio_path, sr=None)

    for index, row in annotations.iterrows():
        if row['syll'] != 'SIL':  # Ignorer les silences
            syllable = extract_syllable(y, sr, row['start'], row['end'])
            mfccs = extract_mfcc(syllable, sr)
            features.append(mfccs)
            syllable_labels.append(row['syll'])

    return features, syllable_labels

def main(directory_path):
    """Parcourir un répertoire pour traiter tous les fichiers CSV et WAV."""
    all_features = []
    all_labels = []

    # Lister tous les fichiers CSV et WAV dans le répertoire
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    wav_files = glob.glob(os.path.join(directory_path, '*.wav'))

    # Créer un dictionnaire des fichiers CSV et WAV
    csv_dict = {os.path.basename(f).replace('.csv', ''): f for f in csv_files}
    wav_dict = {os.path.basename(f).replace('.wav', ''): f for f in wav_files}

    # Trouver les paires correspondantes de fichiers CSV et WAV
    common_keys = set(csv_dict.keys()).intersection(set(wav_dict.keys()))

    for key in common_keys:
        annotation_path = csv_dict[key]
        audio_path = wav_dict[key]

        print(f"Traitement de {audio_path} et {annotation_path}")

        features, syllable_labels = process_files(audio_path, annotation_path)
        all_features.extend(features)
        all_labels.extend(syllable_labels)

    # Appliquer UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    X = np.array(all_features)
    X_umap = reducer.fit_transform(X)

    # Convertir les labels des syllabes en indices pour la coloration
    unique_labels = list(set(all_labels))
    label_indices = [unique_labels.index(label) for label in all_labels]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=label_indices, cmap='Spectral', s=10, alpha=0.7)

    plt.legend(handles=scatter.legend_elements()[0], labels=unique_labels, title="Syllabes")
    plt.title('Projection UMAP de Marron1')
    plt.savefig(output_file, format='png', dpi=300)
    plt.show()

directory_path = 'Dataset_Marron1 PATH'
output_file = 'umap_projection.png'
main(directory_path)