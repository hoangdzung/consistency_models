import numpy as np
import os
import glob
import sys 

def load_npz_files(directory):
    """Load all .npz files in the given directory."""
    npz_files = glob.glob(os.path.join(directory, 'stable*.npz'))
    all_latents = []
    all_labels = []
    
    for file in npz_files:
        data = np.load(file)
        all_latents.extend(data['x_T'])
        all_labels.extend(data['classes'])
    return all_latents, all_labels

def merge_npz_files(directories, output_file):
    """Merge .npz files from multiple directories into a single .npz file."""
    all_latents = []
    all_labels = []

    for directory in directories:
        latents, labels = load_npz_files(directory)
        all_latents.extend(latents)
        all_labels.extend(labels)

    print(len(all_labels))
    all_latents = np.stack(all_latents)
    all_labels = np.array(all_labels)
    
    np.savez(output_file, x_T=all_latents, classes=all_labels)

# List of directories containing the npz files
directories = sys.argv[2:]
output_file = sys.argv[1]

merge_npz_files(directories, output_file)
print(f'Merged npz files saved to {output_file}')