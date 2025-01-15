import sys
import pandas as pd
from ete3 import PhyloTree
import os
sys.path.append('./')
from src.global_config import get_config_WGS_Phlyogeny

def main():

    args = get_config_WGS_Phlyogeny()


    csv_file_path = args.c
    df = pd.read_csv(csv_file_path)
    feature_columns = df.columns[1:-1].tolist()


    nwk_file_path = args.t
    tree = PhyloTree(nwk_file_path, format=2)


    existing_features = []
    for feature in feature_columns:
        nodes = tree.search_nodes(name=feature)
        if len(nodes) == 1:
            existing_features.append(nodes[0])
        elif len(nodes) > 1:

            existing_features.append(nodes[0])
        else:

            print(f"Feature not found in tree: {feature}")


    tree.prune(existing_features)


    output_nwk_file_path = os.path.join(args.o, "phylogeny.nwk")


    tree.write(outfile=output_nwk_file_path)

    print(f"Pruned tree saved to {output_nwk_file_path}")


if __name__ == '__main__':
    main()
