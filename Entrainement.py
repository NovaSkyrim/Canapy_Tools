from canapy import Corpus
from canapy.annotator import SynAnnotator

corpus = Corpus.from_directory(
  audio_directory="/home/utilisateur/Documents/Canapy/Marron1Set",
  annots_directory="/home/utilisateur/Documents/Canapy/Marron1Set",
  annot_format="marron1csv",
  audio_ext=".wav",
)

print(corpus.dataset)

annotator = SynAnnotator()

annotator.fit(corpus)

print(annotator.vocab)

annotator.to_disk("save_directory/annotator")
