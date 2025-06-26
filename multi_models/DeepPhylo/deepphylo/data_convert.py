import os
import sys
import numpy as np
import biom
import argparse

def biom_to_npy(input_file, output_dir):
    # Load BIOM file
    try:
        table = biom.load_table(input_file)
    except Exception as e:
        print(f"Error loading BIOM file: {e}")
        sys.exit(1)

    # Convert BIOM table to NumPy array
    data = table.matrix_data.toarray().T

    # Define output file path
    output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.biom', '.npy'))

    # Save the NumPy array to an .npy file
    try:
        np.save(output_file, data)
        print(f"File successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving npy file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a BIOM file to a NumPy npy file.')
    parser.add_argument('--input_file', type=str, help='Path to the input BIOM file.')
    parser.add_argument('--output_dir', type=str, help='Directory where the output npy file will be saved.')

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Convert and save the file
    biom_to_npy(args.input_file, args.output_dir)
