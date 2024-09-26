import sys
sys.path.append('chemin/vers/canapy')
from canapy import Corpus
from canapy.annotator import SynAnnotator
from canapy.annotator import Annotator

dataset_path = "chemin/vers/dataset"
output_path = "chemin/vers/output"
save_model_path = "chemin/vers/modele/sauvegarde"

if __name__ == '__main__':

  corpus = Corpus.from_directory(
    audio_directory=dataset_path,
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