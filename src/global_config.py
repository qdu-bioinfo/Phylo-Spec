import argparse


# train_test
def get_config_train_test():
    parser = argparse.ArgumentParser(description="Configuration for training and testing the model.")

    parser.add_argument('--PhyloSpec', type=str, required=True, choices=['train', 'test' , 'cv'],
                        help="Mode of operation: 'train' , 'test' or 'cv'.")
    parser.add_argument('-c', type=str, required=True, help="Path to the CSV file (train , test or cv).")
    parser.add_argument('-bs', type=int, default=8, help="Batch size.")
    parser.add_argument('-ep', type=int, default=10, help="Number of epochs.")
    parser.add_argument('-lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('-ch', type=int, default=16, help="Number of channels in the model.")
    parser.add_argument('-ks', type=int, default=3, help="Kernel size for convolution layers.")
    parser.add_argument('-t', type=str, required=True, help="Path to the Newick format tree file.")
    parser.add_argument('-o', type=str, default='./output/',help="Directory to save model and features. Default is './output/'.")
    parser.add_argument('-taxo', type=str, help="Path to the taxonomy file.")
    # os.path.join(os.getcwd(), 'output') or './output/'
    return parser.parse_args()


# 16S_Phlyogeny
def get_config_16S_Phlyogeny():
    parser = argparse.ArgumentParser(description="Process OTU sequences and generate phylogenetic tree.")
    parser.add_argument('-c', type=str, required=True, help="Path to the CSV file(species table) with feature names")
    parser.add_argument('-f',  type=str, required=True, help="Path to the FASTA file(database) with sequences")
    parser.add_argument('-o', type=str, default='./output/', help="Directory to save model and features. Default is './output/'.")
    return parser.parse_args()


# WGS_Phlyogeny
def get_config_WGS_Phlyogeny():
    parser = argparse.ArgumentParser(description="Process feature table and generate pruned phylogenetic tree.")
    parser.add_argument('-c',type=str, required=True, help="Path to the CSV file with feature names")
    parser.add_argument('-t',type=str, required=True, help="Path to the phylogenetic tree (nwk format)")
    parser.add_argument('-o', type=str, default='./output/', help="Directory to save model and features. Default is './output/'.")
    return parser.parse_args()
