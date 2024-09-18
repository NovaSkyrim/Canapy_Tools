import sys
sys.path.append('D:/Inria/canapy')
from canapy import Corpus
from canapy.annotator import SynAnnotator
from canapy.annotator import Annotator
import pandas as pd

dataset_path = "D:/Inria/Datasets/gy6or6_dataset/gy6or6_marron1"
output_path = "output"
save_model_path = "output"
test_set_csv = "D:/Inria/Experiments/DATASIZE/Tests_datasize/1/test_set.csv"
train_set_csv = "D:/Inria/Experiments/DATASIZE/Tests_datasize/1/train_set.csv"
audio_directory = "D:/Inria/Datasets/gy6or6_dataset/gy6or6_Audios"

# Charger les fichiers CSV
test_set_df = pd.read_csv(test_set_csv)
train_set_df = pd.read_csv(train_set_csv)

# Extraire les seqid uniques
test_set_seqs = test_set_df['seqid'].unique()
train_set_seqs = train_set_df['seqid'].unique()

if __name__ == '__main__':

  corpus = Corpus.from_directory(
      audio_directory=audio_directory,
      annots_directory=dataset_path,
      annot_format="marron1csv",
      audio_ext=".wav",
  )

  print(corpus.dataset)

  annotator = SynAnnotator()

  annotator.fit(corpus)

  print(annotator.vocab)

  annotator.to_disk(save_model_path)

  audio_directory = dataset_path

  corpus = Corpus.from_directory(audio_directory=audio_directory)

  annotator = Annotator.from_disk(save_model_path)

  corpus_avec_annotations = annotator.predict(corpus)

  corpus_avec_annotations.to_directory(output_path)

  print("Annotations terminées et enregistrées dans", output_path)