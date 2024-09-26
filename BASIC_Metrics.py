import shutil
import pandas as pd
import numpy as np
import glob
import os
from Levenshtein import distance as levenshtein_distance

def create_frame_annotations(csv_path, sampling_rate, hop_length):
    # Create frame annotations from CSV files
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'start' not in df.columns or 'end' not in df.columns:
        print(f"The columns 'start' or 'end' are missing in the folder: {csv_path}")
        return pd.DataFrame()
    return df

def as_frame_comparison(gold_df, pred_df, hop_length, sampling_rate):
    # Compare frames between gold and predicted data
    gold_df.columns = gold_df.columns.str.strip()
    pred_df.columns = pred_df.columns.str.strip()
    try:
        gold_df["start_frame"] = (gold_df["start"] * sampling_rate / hop_length).astype(int)
        gold_df["end_frame"] = (gold_df["end"] * sampling_rate / hop_length).astype(int)
        pred_df["start_frame"] = (pred_df["start"] * sampling_rate / hop_length).astype(int)
        pred_df["end_frame"] = (pred_df["end"] * sampling_rate / hop_length).astype(int)
    except KeyError as e:
        print(f"Error while comparing frames: {e}")
        return pd.DataFrame()

    gold_frames = []
    pred_frames = []

    for wav in gold_df['wave'].unique():
        gold_wav_df = gold_df[gold_df['wave'] == wav]
        pred_wav_df = pred_df[pred_df['wave'] == wav]

        max_frames = int(np.ceil(gold_wav_df['end'].max() * sampling_rate / hop_length))

        gold_labels = np.full(max_frames, "SIL", dtype=object)
        pred_labels = np.full(max_frames, "SIL", dtype=object)

        for _, row in gold_wav_df.iterrows():
            gold_labels[row['start_frame']:row['end_frame']] = row['syll']

        for _, row in pred_wav_df.iterrows():
            pred_labels[row['start_frame']:row['end_frame']] = row['syll']

        gold_frame_df = pd.DataFrame({'frame': np.arange(max_frames), 'label': gold_labels, 'wave': wav})
        pred_frame_df = pd.DataFrame({'frame': np.arange(max_frames), 'label': pred_labels, 'wave': wav})

        gold_frames.append(gold_frame_df)
        pred_frames.append(pred_frame_df)

    gold_frames = pd.concat(gold_frames)
    pred_frames = pd.concat(pred_frames)

    comparison_df = pd.merge(gold_frames, pred_frames, on=['frame', 'wave'], suffixes=('_gold', '_pred'))

    incorrect_frames = comparison_df[comparison_df['label_gold'] != comparison_df['label_pred']]

    return comparison_df

def calculate_frame_error_rate(comparison_df):
    # Calculate the frame error rate by comparing gold and predicted frames
    total_frames = len(comparison_df)
    incorrect_frames = np.sum(comparison_df['label_gold'] != comparison_df['label_pred'])
    return incorrect_frames / total_frames if total_frames > 0 else float('nan')

def compute_frame_error_rate_statistics(gold_folder, pred_folder, sampling_rate, hop_length):
    # Calculate the frame error rate between two folders of annotations
    gold_files = sorted(os.listdir(gold_folder))
    pred_files = sorted(os.listdir(pred_folder))

    # Filter CSV files
    gold_files = [f for f in gold_files if f.endswith('.csv')]
    pred_files = [f for f in pred_files if f.endswith('.csv')]

    # Ensure that the files match
    if set(gold_files) != set(pred_files):
        raise ValueError("Files don't match between the two folders.")

    frame_error_rates = []

    for file in gold_files:
        gold_path = os.path.join(gold_folder, file)
        pred_path = os.path.join(pred_folder, file)

        gold_df = create_frame_annotations(gold_path, sampling_rate, hop_length)
        pred_df = create_frame_annotations(pred_path, sampling_rate, hop_length)

        if not gold_df.empty and not pred_df.empty:
            comparison_df = as_frame_comparison(gold_df, pred_df, hop_length, sampling_rate)
            if not comparison_df.empty:
                frame_error_rate = calculate_frame_error_rate(comparison_df)
                frame_error_rates.append(frame_error_rate)

    if frame_error_rates:
        average_frame_error_rate = np.mean(frame_error_rates)
        std_frame_error_rate = np.std(frame_error_rates)
    else:
        print("The Frame Error Rate cannot be calculated.")
        average_frame_error_rate = float('nan')
        std_frame_error_rate = float('nan')

    return average_frame_error_rate, std_frame_error_rate

def extract_labels_from_csv_files(directory):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    csv_files.sort()
    all_labels = []

    for file in csv_files:
        df = pd.read_csv(file)
        if 'syll' in df.columns:
            labels = df['syll']
        elif 'label' in df.columns:
            labels = df['label']
        else:
            print(f"The column 'syll' or 'label' does not exist in the file {file}")
            continue

        filtered_labels = labels[(labels != 'SIL') & (labels != 'TRASH')]
        all_labels.extend(map(str, filtered_labels))

    return all_labels

def compare_labels(directory_path, reference_file_path):
    labels_list_from_directory = remove_consecutive_duplicates(extract_labels_from_csv_files(directory_path))
    labels_list_from_reference = remove_consecutive_duplicates(extract_labels_from_csv_files(reference_file_path))

    distance = levenshtein_distance(labels_list_from_directory, labels_list_from_reference)
    # The distance represents the number of edits (insertions and deletions) needed for the two lists to be identical

    syllable_error_rate = (distance / len(labels_list_from_reference)) * 100
    # The syllable error rate is normalized and expressed as a percentage

    return distance, syllable_error_rate

def remove_consecutive_duplicates(s):
    result = s[0] if s else ""
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:
            result += s[i]

    return result


dataset_path = "/home/user/Documents/Canapy/Datasets/Marron1Full"  # Path to the folder containing the original annotations
working_directory = "Path/To/Your/Working/Directory"
bird_name = "Marron1"  # Name of the working folder
seeds_values = list(range(1, 11))  # Values of seeds used in the Basic
sampling_rate = 44100  # Specify the sampling rate of the audio
hop_length = 512  # Specify the hop_length of the audio


results = []
for i in ['Test', 'Train']:
    if i == 'Test':
        print("TEST STATS\n")
    elif i == 'Train':
        print("TRAIN STATS\n")

    for seed in seeds_values:
        directory_path = f'{working_directory}/{bird_name}/{seed}/{i}_set'
        reference_folder_path = f'{working_directory}/{bird_name}/{seed}/{i}_correction'

        distance, syllable_error_rate = compare_labels(directory_path, reference_folder_path)
        average_frame_error_rate, std_frame_error_rate = compute_frame_error_rate_statistics(reference_folder_path, directory_path, sampling_rate, hop_length)

        print(f"For seed {seed} of the syntactic model:")
        print(f"Frame Error Rate mean: {average_frame_error_rate:.4f}")
        print(f"Frame Error Rate std_dev: {std_frame_error_rate:.4f}")
        print(f"The edit distance between the two strings is: {distance}")
        print(f"The syllable error rate between the two strings is: {round(syllable_error_rate, 2)}%\n")

        # Add results to the list
        results.append({
            'Dataset': i,
            'Seed': seed,
            'Syllable Error Rate': syllable_error_rate,
            'Average Frame Error Rate': average_frame_error_rate,
            'STD Frame Error Rate': std_frame_error_rate
        })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Save to a CSV file
results_df.to_csv(f'{working_directory}/{bird_name}/results_summary.csv', index=False)
print(f"Metrics saved to results_summary.csv")
