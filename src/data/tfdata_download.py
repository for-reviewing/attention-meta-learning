# src/data/tfdata_download.py

'''
Note: After downloading the dataset, the contents of the folder 
"ZIP.data.mend.com_publ-file_data_tywb_file_d565-c1rDQyRTmE0CqGGXmH53WlQp0NWefMfDW89aj1A0m5D_A/"
are manually moved to the parent "extracted/" folder, and the original folder is deleted.

New folder structure will be as follow:
├── raw/
│   └── plant_village/
│       ├── downloads/
│       │   ├── extracted/
│       │   │   └── Plant_leave_diseases_dataset_without_augmentation/
'''


import tensorflow_datasets as tfds
import os

# Directory where the Plant Village dataset will be stored
pv_dataset_dir = 'data/raw/plant_village'

# Check if directory exists and is not empty to avoid redundant downloads
if os.path.exists(pv_dataset_dir) and os.listdir(pv_dataset_dir):
    print("Dataset already exists. Skipping download.")
else:
    # Create directory if it doesn't exist
    os.makedirs(pv_dataset_dir, exist_ok=True)
    
    # Download and save the dataset
    dataset, info = tfds.load('plant_village', with_info=True, as_supervised=True, data_dir=pv_dataset_dir)
    print("Dataset downloaded successfully.")