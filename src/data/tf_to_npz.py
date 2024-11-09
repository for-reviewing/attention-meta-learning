# src/data/tf_to_npz.py
# This script needs at least around 32 GB RAM and 7.4 GB disk space

import tensorflow_datasets as tfds
import numpy as np
import os

# Path to the NPZ file
np_data_dir = 'data/interim/'
npz_file_path = os.path.join(np_data_dir, 'plant_village_data.npz')

# Check if the NPZ file already exists
if os.path.exists(npz_file_path):
    print(f"Data already exists at {npz_file_path}. Skipping conversion.")
else:
    # Create the interim data directory if it doesn't exist
    os.makedirs(np_data_dir, exist_ok=True)

    # Load the dataset from TFDS
    dataset_dir = 'data/raw/plant_village'
    dataset, info = tfds.load('plant_village', split='train', with_info=True, data_dir=dataset_dir)

    # Function to convert TFDS dataset to NumPy arrays
    def tfds_to_numpy(dataset):
        images = []
        labels = []
        for example in tfds.as_numpy(dataset):
            images.append(example['image'])
            labels.append(example['label'])
        return np.array(images), np.array(labels)

    # Convert the TFDS dataset to NumPy arrays
    images, labels = tfds_to_numpy(dataset)

    # Save the NumPy arrays to an NPZ file
    np.savez_compressed(npz_file_path, images=images, labels=labels)

    print(f"Data has been saved to {npz_file_path}")
