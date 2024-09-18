import numpy as np
import pandas as pd
import os

def create_frame_annotations(csv_path, sampling_rate, hop_length):
    df = pd.read_csv(csv_path)
    
    # Vérifier et nettoyer les colonnes
    df.columns = df.columns.str.strip()  # Supprime les espaces autour des noms de colonnes
    print(f"Colonnes après nettoyage dans {csv_path}: {df.columns}")
    
    # Vérifier la présence des colonnes 'start' et 'end'
    if 'start' not in df.columns or 'end' not in df.columns:
        print(f"Les colonnes 'start' ou 'end' sont manquantes dans le fichier : {csv_path}")
        return pd.DataFrame()
    
    return df

def as_frame_comparison(gold_df, pred_df, hop_length, sampling_rate):
    """Étendre gold_df au format frame pour comparaison avec pred_df."""
    
    # Nettoyer les colonnes si nécessaire
    gold_df.columns = gold_df.columns.str.strip()
    pred_df.columns = pred_df.columns.str.strip()
    
    # Vérifier la présence des colonnes nécessaires
    try:
        gold_df["start_frame"] = (gold_df["start"] * sampling_rate / hop_length).astype(int)
        gold_df["end_frame"] = (gold_df["end"] * sampling_rate / hop_length).astype(int)
        pred_df["start_frame"] = (pred_df["start"] * sampling_rate / hop_length).astype(int)
        pred_df["end_frame"] = (pred_df["end"] * sampling_rate / hop_length).astype(int)
    except KeyError as e:
        print(f"Erreur lors de la comparaison des frames : {e}")
        print("Contenu de gold_df :")
        print(gold_df.head())
        print("Colonnes présentes dans gold_df :", gold_df.columns)
        print("La comparaison n'a pas pu être effectuée.")
        return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur
    
    gold_frames = []
    pred_frames = []
    
    for wav in gold_df['wave'].unique():
        print(f"Traitement du fichier audio : {wav}")

        gold_wav_df = gold_df[gold_df['wave'] == wav]
        pred_wav_df = pred_df[pred_df['wave'] == wav]
        
        print(f"Annotations orales : {len(gold_wav_df)}")
        print(gold_wav_df)
        print(f"Annotations prédites : {len(pred_wav_df)}")
        print(pred_wav_df)
        
        max_frames = int(np.ceil(gold_wav_df['end'].max() * sampling_rate / hop_length))
        print(f"Nombre total de frames pour {wav} : {max_frames}")

        gold_labels = np.full(max_frames, "SIL", dtype=object)
        pred_labels = np.full(max_frames, "SIL", dtype=object)
        
        for _, row in gold_wav_df.iterrows():
            print(f"Annotation orales pour la frame de {row['start_frame']} à {row['end_frame']}: {row['syll']}")
            gold_labels[row['start_frame']:row['end_frame']] = row['syll']
        
        for _, row in pred_wav_df.iterrows():
            print(f"Annotation prédites pour la frame de {row['start_frame']} à {row['end_frame']}: {row['syll']}")
            pred_labels[row['start_frame']:row['end_frame']] = row['syll']
        
        gold_frame_df = pd.DataFrame({'frame': np.arange(max_frames), 'label': gold_labels, 'wave': wav})
        pred_frame_df = pd.DataFrame({'frame': np.arange(max_frames), 'label': pred_labels, 'wave': wav})
        
        print(f"Frames orales pour {wav} :")
        print(gold_frame_df.head())
        print(f"Frames prédites pour {wav} :")
        print(pred_frame_df.head())
        
        gold_frames.append(gold_frame_df)
        pred_frames.append(pred_frame_df)
    
    gold_frames = pd.concat(gold_frames)
    pred_frames = pd.concat(pred_frames)
    
    print(f"Nombre total de frames orales après concaténation : {len(gold_frames)}")
    print(gold_frames.head())
    print(f"Nombre total de frames prédites après concaténation : {len(pred_frames)}")
    print(pred_frames.head())
    
    comparison_df = pd.merge(gold_frames, pred_frames, on=['frame', 'wave'], suffixes=('_gold', '_pred'))
    
    print(f"Nombre de frames après comparaison : {len(comparison_df)}")
    print(comparison_df.head())
    
    # Identifier les frames incorrectes
    incorrect_frames = comparison_df[comparison_df['label_gold'] != comparison_df['label_pred']]
    if not incorrect_frames.empty:
        print(f"Frames incorrectes : {len(incorrect_frames)}")
        print(incorrect_frames)
    
    return comparison_df

def calculate_frame_error_rate(comparison_df):
    total_frames = len(comparison_df)
    incorrect_frames = np.sum(comparison_df['label_gold'] != comparison_df['label_pred'])
    
    print(f"Total des frames comparées : {total_frames}")
    print(f"Nombre de frames incorrectes : {incorrect_frames}")
    
    return incorrect_frames / total_frames if total_frames > 0 else float('nan')

# Paramètres
sampling_rate = 44100  # Exemple: taux d'échantillonnage de 44,1 kHz
hop_length = 512       # Exemple: hop length de 512 échantillons

# Charger et préparer les annotations
gold_annotations_path = '/home/utilisateur/Documents/Canapy/Datasets/Marron1Full/130_marron1_May_26_2016_23553363.csv'
pred_annotations_path = '/home/utilisateur/Documents/Canapy/Experiments/BASIC/Marron1Full_phrase/100_marron1_May_24_2016_62101389.csv'

# Afficher les 5 premières lignes et les colonnes des CSVs
gold_df = pd.read_csv(gold_annotations_path, header=0, skipinitialspace=True)
pred_df = pd.read_csv(pred_annotations_path, header=0, skipinitialspace=True)

print("Colonnes du Gold CSV :", gold_df.columns)
print(gold_df.head())

print("Colonnes du Pred CSV :", pred_df.columns)
print(pred_df.head())

# Créer les annotations de frames
gold_annotations = create_frame_annotations(gold_annotations_path, sampling_rate, hop_length)
pred_annotations = create_frame_annotations(pred_annotations_path, sampling_rate, hop_length)

# Comparer les annotations au niveau des frames
comparison_df = as_frame_comparison(gold_annotations, pred_annotations, hop_length, sampling_rate)

# Calculer le Frame Error Rate (FER)
if not comparison_df.empty:
    frame_error_rate = calculate_frame_error_rate(comparison_df)
    print(f"Frame Error Rate: {frame_error_rate:.4f}")
else:
    print("La comparaison n'a pas pu être effectuée.")

