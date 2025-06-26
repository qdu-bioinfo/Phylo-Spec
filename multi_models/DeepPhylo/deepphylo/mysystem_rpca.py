import biom
import pandas as pd
import numpy as np
from Bio import Phylo
from skbio import TreeNode
from multi_models.DeepPhylo.deepphylo.preprocessing import fast_unifrac
from multi_models.DeepPhylo.deepphylo.plot import reducer


def import_and_process_data(table_path, tree_path):

    df = pd.read_csv(table_path, sep=',')
    sample_ids = df.iloc[:, 0].values
    feature_data = df.iloc[:, 1:-1]

    bt = biom.Table(feature_data.T.values, observation_ids=feature_data.columns, sample_ids=sample_ids)


    tntree = Phylo.read(tree_path, "newick")


    target_features = set(bt.ids('observation'))
    for clade in tntree.get_terminals():
        if clade.name not in target_features:
            tntree.prune(clade)

    return bt, tntree


def compute_distance_matrix(bt, tntree):
    # Compute pairwise distances between features using the phylogenetic tree
    fid_seq_list = list(bt.ids('observation'))
    distance_matrix = np.zeros((len(fid_seq_list), len(fid_seq_list)))
    terminals = tntree.get_terminals()

    for node1 in terminals:
        if node1.name not in fid_seq_list:
            continue
        i = fid_seq_list.index(node1.name)
        for node2 in terminals:
            if node2.name not in fid_seq_list:
                continue
            j = fid_seq_list.index(node2.name)
            distance_matrix[i, j] = tntree.distance(node1, node2)

    return distance_matrix


def compute_embedding(distance_matrix):

    n_components = min(200, min(distance_matrix.shape))


    otu_evol_embedding = reducer(distance_matrix, 'pca', n_components)

    # output
    np.save(r'.npy', otu_evol_embedding)

    return otu_evol_embedding


if __name__ == "__main__":
    # input
    table_path = r"csv"
    tree_path = r"nwk"

    # Import and process data
    bt, tntree = import_and_process_data(table_path, tree_path)

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(bt, tntree)

    # Compute embedding
    otu_evol_embedding = compute_embedding(distance_matrix)

    print("Analysis complete. Distance matrix and embedding saved.")

