import scipy.io
import numpy as np

# Load .mat file
data = scipy.io.loadmat('llb11_annotation_Apr_2019_Vika_4TF.mat')

# Access the 'elements' field
elements = data['elements']

# Check the type and shape of 'elements'
print("Type of 'elements':", type(elements))
print("Shape of 'elements':", elements.shape)

# Process each record
for record in elements[0]:
    filenum = record['filenum'][0]
    start_times = record['segFileStartTimes'][0]
    end_times = record['segFileEndTimes'][0]
    seg_type = record['segType'][0]

    # Calculate durations
    durations = end_times - start_times

    # Print or process the data as needed
    print(f"File Number: {filenum}")
    print(f"Start Times: {start_times}")
    print(f"End Times: {end_times}")
    print(f"Segment Types: {seg_type}")
    print(f"Durations: {durations}")
    print('-' * 40)
