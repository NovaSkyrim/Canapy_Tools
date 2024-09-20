import random
import sys
sys.path.append('D:/Inria/canapy')
import pandas as pd
from canapy import Corpus
from canapy.annotator import SynAnnotator
from canapy.annotator import Annotator

def update_train_column(df, num_files):
    
    unique_files = df['annot_path'].unique()
    
    if num_files > len(unique_files):
        raise ValueError(f"Le nombre de fichiers demandés ({num_files}) est supérieur au nombre de fichiers disponibles ({len(unique_files)}).")
    
    selected_files = random.sample(list(unique_files), num_files)

    df = df.assign(train=False)
    
    df['train'] = df['annot_path'].apply(lambda x: True if x in selected_files else False)
    
    df.to_csv("D:/Inria/Experiments/DATASIZE/Gy6or6_DATASIZE_10iter/1/30/Split.csv", index=False)
    
    return df

dataset_path = "D:/Inria/Datasets/gy6or6_dataset/gy6or6_marron1"
output_path = "D:/Inria/Experiments/DATASIZE/Marron1_DATASIZE_10iter/1/30/Annots"
save_model_path = "D:/Inria/Experiments/DATASIZE/Marron1_DATASIZE_10iter/1/30/model"
audio_directory = "D:/Inria/Datasets/gy6or6_dataset/gy6or6_Audios"

if __name__ == '__main__':

  corpus = Corpus.from_directory(
      audio_directory=audio_directory,
      annots_directory=dataset_path,
      annot_format="marron1csv",
      audio_ext=".wav",
  )

  print(corpus.dataset)

  annotator = SynAnnotator()

  corpus.dataset = update_train_column(corpus.dataset, 30)

  annotator.fit(corpus)

  print(annotator.vocab)

  annotator.to_disk(save_model_path)

  audio_directory = dataset_path

  corpus = Corpus.from_directory(audio_directory=audio_directory)

  annotator = Annotator.from_disk(save_model_path)

  corpus_avec_annotations = annotator.predict(corpus)

  corpus_avec_annotations.to_directory(output_path)

  print("Annotations terminées et enregistrées dans", output_path)