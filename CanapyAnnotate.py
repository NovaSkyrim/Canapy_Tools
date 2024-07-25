from canapy import Corpus
from canapy.annotator import Annotator

# Chargez le corpus audio non annoté
audio_directory = "C:/Users/clemd/OneDrive/Bureau/CanapyNathan/Marron1FullAudios"
corpus = Corpus.from_directory(audio_directory=audio_directory)

# Chargez le modèle annotateur entraîné
chemin_vers_modele = "C:/Users/clemd/OneDrive/Bureau/CanapyNathan/Marron1SetOutput/output/model/syn-esn"
annotator = Annotator.from_disk(chemin_vers_modele)

# Annoter le corpus
corpus_avec_annotations = annotator.predict(corpus)

# Enregistrer les annotations
output_directory = "Marron1SetAudioOutput"
corpus_avec_annotations.to_directory(output_directory)

print("Annotations terminées et enregistrées dans", output_directory)