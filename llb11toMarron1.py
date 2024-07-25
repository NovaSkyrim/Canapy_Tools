import scipy.io
import numpy as np

data = scipy.io.loadmat('Chemin vers le fichier .mat')

elements = data['elements']

print("Type of 'elements':", type(elements))
print("Shape of 'elements':", elements.shape)

for record in elements[0]:
    filenum = record['filenum'][0]
    start_times = record['segFileStartTimes'][0]
    end_times = record['segFileEndTimes'][0]
    seg_type = record['segType'][0]

    durations = end_times - start_times

    print(f"File Number: {filenum}")
    print(f"Start Times: {start_times}")
    print(f"End Times: {end_times}")
    print(f"Segment Types: {seg_type}")
    print(f"Durations: {durations}")
    print('-' * 40)
