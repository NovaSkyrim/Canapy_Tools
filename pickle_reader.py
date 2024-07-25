from canapy.annotator import Annotator
from config import default_config

annotator = Annotator.from_disk('syn-esn', config=default_config)

print(annotator.config)