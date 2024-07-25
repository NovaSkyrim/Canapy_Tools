from canapy import Corpus
from canapy.annotator import Annotator

audio_directory = "Chemin vers le dossier des audios non annotés"
corpus = Corpus.from_directory(audio_directory=audio_directory)

chemin_vers_modele = "Chemin vers le modèle"
annotator = Annotator.from_disk(chemin_vers_modele)

corpus_avec_annotations = annotator.predict(corpus)

output_directory = "Output"
corpus_avec_annotations.to_directory(output_directory)

print("Annotations terminées et enregistrées dans", output_directory)