import os
import sys
import numpy as np
from scipy.io import savemat

def convert_npy_to_mat(npy_file, output_dir=None):
    """
    Convert a .npy file to a .mat file.

    Parameters:
        npy_file (str): Path to the .npy file.
        output_dir (str): Directory to save the .mat file (optional).
        
    Examples:
        python npy2mat.py <path_to_npy_file> [optional_output_directory]
        python npy2mat.py example.npy output (Convert example.npy to output/example.mat)
    """
    # Load the .npy file
    try:
        array = np.load(npy_file)
    except Exception as e:
        print(f"Error loading {npy_file}: {e}")
        return

    # Create .mat filename
    base_name = os.path.basename(npy_file)
    mat_file_name = os.path.splitext(base_name)[0] + '.mat'

    # Set output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        mat_file_path = os.path.join(output_dir, mat_file_name)
    else:
        mat_file_path = mat_file_name

    # Save the array as a .mat file
    try:
        savemat(mat_file_path, {os.path.splitext(base_name)[0]: array})
        print(f"Converted {npy_file} to {mat_file_path}")
    except Exception as e:
        print(f"Error saving {mat_file_path}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python npy2mat.py <npy_file> [output_dir]")
        sys.exit(1)

    npy_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.isfile(npy_file):
        print(f"Error: {npy_file} does not exist or is not a file.")
        sys.exit(1)

    convert_npy_to_mat(npy_file, output_dir)

if __name__ == "__main__":
    main()