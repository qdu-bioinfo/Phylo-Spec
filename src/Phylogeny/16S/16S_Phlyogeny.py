import subprocess
import sys
from Bio import SeqIO
import pandas as pd
import os
sys.path.append('./')
from src.global_config import get_config_16S_Phlyogeny

def main():
    args = get_config_16S_Phlyogeny()


    script_dir = os.path.dirname(os.path.realpath(__file__))


    csv_file_path = args.c
    fasta_file_path = args.f
    output_dir = args.o


    data = pd.read_csv(csv_file_path)


    feature_names = data.columns[1:-1].tolist()

    sequences = SeqIO.to_dict(SeqIO.parse(fasta_file_path, 'fasta'))


    selected_sequences = []
    not_found_otus = []

    for feature_name in feature_names:
        if feature_name in sequences:
            selected_sequences.append(sequences[feature_name])
        else:
            not_found_otus.append(feature_name)


    output_fasta_file = os.path.join(output_dir, 'phylogeny.fasta')
    SeqIO.write(selected_sequences, output_fasta_file, 'fasta')
    print(f"Successfully saved sequence to {output_fasta_file}")

    if not_found_otus:
        print("The following OTU numbers were not found in the database:")
        for otu in not_found_otus:
            print(otu)
    else:
        print("All OTU numbers were found and extracted.")


    print("\n*** Start: [Sequence Align with Mafft and run FastTree]")


    mafft_path = os.path.join(script_dir, 'mafft-win', 'mafft.bat')
    fasttree_path = os.path.join(script_dir, 'FastTree.exe')


    mafft_output_file = output_fasta_file.replace(".fasta", "-align.fasta")
    mafft_cmd = f"call {mafft_path} --auto {output_fasta_file} > {mafft_output_file}"
    subprocess.call(mafft_cmd, shell=True)


    fasttree_output_file = mafft_output_file.replace("-align.fasta", ".nwk")
    bootstrap_replicates = 1000
    fasttree_cmd = f"{fasttree_path} -gtr -nt -boot {bootstrap_replicates} {mafft_output_file} > {fasttree_output_file}"
    subprocess.call(fasttree_cmd, shell=True)

    print(f"Alignment and phylogenetic tree generation completed.")
    print(f"Mafft output file: {mafft_output_file}")
    print(f"FastTree output file: {fasttree_output_file}")


if __name__ == '__main__':
    main()
