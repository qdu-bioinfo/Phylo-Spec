import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
def load_and_preprocess_data(csv_path, tree):
    data = pd.read_csv(csv_path)
    leaf_names = [leaf.name for leaf in tree.get_terminals()]
    matched_columns = [col for col in leaf_names if col in data.columns]
    remaining_columns = [col for col in data.columns[1:-1] if col not in matched_columns]
    ordered_columns = [data.columns[0]] + matched_columns + remaining_columns + [data.columns[-1]]
    data = data[ordered_columns]
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    y_encoded, encoder = encode_labels(y)

    return X, y_encoded, encoder, data

def encode_labels(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder

def match_leaf_nodes(tree, data):
    leaf_to_species = {}
    for leaf in tree.get_terminals():
        leaf_name = leaf.name
        for species in data.columns[1:-1]:
            if species in leaf_name:
                leaf_to_species[leaf_name] = species
                break
    return leaf_to_species

def assign_unique_names(tree):
    internal_node_counter = 0
    for clade in tree.find_clades(order="level"):
        if not clade.is_terminal() and clade.name is None:
            clade.name = f"internal_{internal_node_counter}"
            internal_node_counter += 1
    return tree

def get_conv_order(tree):
    nodes = []
    parents = {}
    conv_order = []
    node_relations = {}

    def postorder_traversal(node, parent, nodes, parents, conv_order):
        if node.is_terminal():
            nodes.append(node.name)
            parents[node.name] = parent
        else:
            children = []
            for child in node:
                postorder_traversal(child, node.name, nodes, parents, conv_order)
                children.append(child.name)
            nodes.append(node.name)
            parents[node.name] = parent
            conv_order.append((children, node.name))
            node_relations[node.name] = children

    postorder_traversal(tree.root, None, nodes, parents, conv_order)
    return nodes, parents, conv_order, node_relations

def calculate_node_weights(tree):
    node_weights = {}
    for clade in tree.find_clades(order="level"):
        if clade.branch_length is not None:
            node_weights[clade.name] = 1 - clade.branch_length
        else:
            node_weights[clade.name] = 1.0
    return node_weights

def save_node_features_with_pickle(node_features, node_relations, node_weights, train_group, output_file):
    node_features_dict = {}
    for node, feature_list in node_features.items():
        concatenated_features = np.concatenate(feature_list, axis=0)
        node_features_dict[node] = concatenated_features
    branch_lengths = {node: weight for node, weight in node_weights.items()}

    combined_data = {
        'features': node_features_dict,
        'branch_lengths': branch_lengths,
        'relations': node_relations,
        'true_labels': train_group
    }

    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)



def process_unclassified_features(tree, abundance_table, taxonomy_path):

    table = abundance_table.copy()
    group_col = table.pop(table.columns[-1])
    taxonomy_table = pd.read_csv(taxonomy_path)

    unclassified_features = [col for col in abundance_table.columns if 'Unclassified' in col or 'unclassified' in col]

    if not unclassified_features:
        table["Group"] = group_col
        return table, tree

    for feature in unclassified_features:
        missing_species = pd.Series(dtype='str')  # 修改变量名为 missing_species

        taxonomic_levels = ['Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom']
        for level in taxonomic_levels:
            matched_taxa = taxonomy_table.loc[taxonomy_table['Species'] == feature, level]
            if not matched_taxa.empty:
                taxon = matched_taxa.values[0]

                missing_species = taxonomy_table[
                    (taxonomy_table[level] == taxon) &
                    (~taxonomy_table['Species'].isin(abundance_table.columns)) &
                    (~taxonomy_table['Species'].str.contains('Unclassified'))
                ]['Species']  # 保持逻辑一致，只是改为 missing_species

                if not missing_species.empty:
                    break

        if missing_species.empty:
            print(f"No missing species found in abundance table for feature '{feature}'.")
            continue

        unclassified_abundance = abundance_table[feature]
        average_abundance = unclassified_abundance / len(missing_species)

        for species in missing_species:  # 修改变量名为 species
            if species in table.columns:
                table[species] += average_abundance
            else:
                table[species] = average_abundance
        table.drop(columns=[feature], inplace=True)

    table["Group"] = group_col

    return table, tree

