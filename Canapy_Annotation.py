from canapy import Corpus
from canapy.annotator import Annotator

audio_directory = "/home/utilisateur/Documents/Canapy/canapy/gy6or6_dataset/Test"
corpus = Corpus.from_directory(audio_directory=audio_directory)

chemin_vers_modele = "/home/utilisateur/Documents/Canapy/canapy/output/model/syn-esn"
annotator = Annotator.from_disk(chemin_vers_modele)

corpus_avec_annotations = annotator.predict(corpus)

output_directory = "gy6or6_output"
corpus_avec_annotations.to_directory(output_directory)

print("Annotations terminées et enregistrées dans", output_directory)
