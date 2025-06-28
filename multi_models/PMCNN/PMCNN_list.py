
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Bio import Phylo
import os


# input
newick_path = r"nwk"
# output
csv_path = r"csv"


def calculate_cophenetic_distances(tree_file, output_file):
    tree = Phylo.read(tree_file, "newick")
    terminals = tree.get_terminals()
    num_terminals = len(terminals)
    matrix = np.zeros((num_terminals, num_terminals))

    # Calculate cophenetic distances
    for i in range(num_terminals):
        for j in range(num_terminals):
            matrix[i, j] = tree.distance(terminals[i], terminals[j])

    df = pd.DataFrame(matrix, index=[term.name for term in terminals], columns=[term.name for term in terminals])
    df.to_csv(output_file)
    print(f"Cophenetic distances calculated and saved to {output_file}")


def drop_row_col(infile, outfile):
    data = pd.read_csv(infile, header=None, skiprows=[0])
    data = data.iloc[:, 1:]
    data.to_csv(outfile, index=False, header=None)


def zero_one(infile):
    data = pd.read_csv(infile, header=None)
    max_value = data.max().max()
    print("Initial maximum value:", max_value)
    min_value = np.exp(-(max_value * max_value) / 0.5)
    print("Transformed minimum value:", min_value)
    min_val = data.min().min()
    print("Initial minimum value:", min_val)
    max_value_transformed = np.exp(-(min_val * min_val) / 0.5)
    print("Transformed maximum value:", max_value_transformed)
    df = np.exp(-(data * data) / 0.5)
    df = df.astype(np.float32)
    df.to_csv(infile, header=None, index=False)
    return max_value_transformed, min_value


def main():
    dist_file = newick_path.replace(".nwk", "_dist.csv")

    calculate_cophenetic_distances(newick_path, dist_file)

    out_file = dist_file.replace("_dist.csv", "_tran.csv")

    drop_row_col(dist_file, out_file)
    max_value, min_value = zero_one(out_file)

    thresholds = [0.1, 0.01, 0.3, 0.2]
    length = len(thresholds)
    intermediate_file = dist_file.replace("_dist.csv", ".npy")
    output_file = dist_file.replace("_dist.csv", "_tran.npy")

    data1 = pd.read_csv(out_file, header=None)
    np.save(intermediate_file, data1)

    def csv_npy(infile, intermediate_file, outfile):
        data = pd.read_csv(infile, header=None)
        np.save(intermediate_file, data)
        data_loaded = np.load(intermediate_file)
        return data_loaded, data.shape[0]

    def hierarchical_clustering(distance_matrix):
        def my_dist(p1, p2):
            return 1.0 - distance_matrix[int(p1), int(p2)]

        X = np.arange(distance_matrix.shape[0]).reshape(-1, 1)
        linked = linkage(X, method='single', metric=my_dist)
        result1 = [[] for _ in range(length)]
        for i, threshold in enumerate(thresholds):
            clusters = fcluster(linked, t=threshold, criterion='distance')
            for j, item in enumerate(clusters):
                result1[i].append(item)
            print(f"Cluster indices at threshold {threshold}:")
            print(result1[i], "\nTotal clusters:", max(result1[i]), "clusters\n")
        return result1

    def index_of_duplicates(arr):
        index_dict = {}
        for i, val in enumerate(arr):
            if val in index_dict:
                index_dict[val].append(i)
            else:
                index_dict[val] = [i]
        return [idx for indexes in index_dict.values() for idx in indexes]

    def get_id():
        last_order = [[] for _ in range(length)]
        for i in range(length):
            last_order[i] = index_of_duplicates(result[i])
        return last_order

    data, data_sum = csv_npy(out_file, intermediate_file, output_file)
    result = hierarchical_clustering(data)
    My_order = get_id()

    My_fea_savefile = csv_path
    data = pd.read_csv(dist_file)
    raw_order = data.iloc[:, 0].tolist()

    def get_fea():
        My_fea = [[] for _ in range(len(My_order_result))]
        for j in range(len(My_order_result)):
            for x in My_order[j]:
                My_fea[j].append(raw_order[x])
        return My_fea

    def save():
        data = pd.DataFrame(My_feature)
        data.to_csv(My_fea_savefile, index=False)
        data1 = pd.read_csv(My_fea_savefile)
        print(data1.head())

    My_order_result = get_id()
    My_feature = get_fea()

    save()


if __name__ == "__main__":
    main()
