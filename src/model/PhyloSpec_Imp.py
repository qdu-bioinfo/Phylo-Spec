
import sys
import torch
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from joblib import Parallel, delayed
from multiprocessing import cpu_count
sys.path.append('./')
from src.global_config import get_config_feature_importance

def normalize_features(node_features):
    scaler = StandardScaler()
    normalized_features = {}
    for node, features in node_features.items():
        if features.dim() == 2:
            flattened_features = features.numpy()
        elif features.dim() > 2:
            flattened_features = features.view(features.size(0), -1).numpy()
        else:
            raise ValueError(f"Unexpected feature dimensions for node {node}: {features.dim()}")

        if flattened_features.size == 0:
            normalized_features[node] = torch.tensor(flattened_features, dtype=torch.float32)
            continue

        normalized = scaler.fit_transform(flattened_features)
        normalized_features[node] = torch.tensor(normalized, dtype=torch.float32)
    return normalized_features


def entropy(y):
    if len(y) == 0:
        return 0.0
    counts = Counter(y)
    probabilities = np.array(list(counts.values())) / len(y)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(X, y, feature_idx, thresholds, parent_entropy):
    info_gains = []
    for thresh in thresholds:
        left_indices = X[:, feature_idx] <= thresh
        right_indices = X[:, feature_idx] > thresh

        y_left = y[left_indices]
        y_right = y[right_indices]

        ent_left = entropy(y_left)
        ent_right = entropy(y_right)

        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        weighted_entropy = 0.0
        if n_left > 0:
            weighted_entropy += (n_left / n) * ent_left
        if n_right > 0:
            weighted_entropy += (n_right / n) * ent_right

        info_gain = parent_entropy - weighted_entropy
        info_gains.append(info_gain)
    return info_gains


def best_information_gain(X, y, feature_idx, parent_entropy):
    sorted_indices = np.argsort(X[:, feature_idx])
    sorted_X = X[sorted_indices, feature_idx]

    unique_values = np.unique(sorted_X)
    if len(unique_values) == 1:
        return 0.0

    thresholds = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

    info_gains = information_gain(X, y, feature_idx, thresholds, parent_entropy)

    return max(info_gains) if info_gains else 0.0


def calculate_feature_information_gain_parallel(node, features, y):
    X = features.numpy()
    if X.shape[0] == 0:
        return (node, np.zeros(X.shape[1]))
    parent_entropy = entropy(y)
    n_features = X.shape[1]
    gains = []
    for feature_idx in range(n_features):
        gain = best_information_gain(X, y, feature_idx, parent_entropy)
        gains.append(gain)
    gains = np.array(gains)
    return (node, gains)


def calculate_feature_information_gain(node_features, true_labels):
    feature_importances = {}
    nodes = list(node_features.keys())

    results = Parallel(n_jobs=cpu_count())(
        delayed(calculate_feature_information_gain_parallel)(node, node_features[node], true_labels) for node in nodes
    )

    for node, gains in results:
        feature_importances[node] = gains
    return feature_importances



def calculate_node_importance_recursive(node, parent_importance, feature_importances, node_relations, branch_lengths,
                                        importance_scores):
    gains = feature_importances.get(node, np.array([]))
    if gains.size == 0:
        node_importance = 0.0
    else:
        node_importance = np.mean(gains)

    current_importance = node_importance + parent_importance
    importance_scores[node] = current_importance

    if node in node_relations:
        for child in node_relations[node]:
            dist = branch_lengths.get(child)
            calculate_node_importance_recursive(child, current_importance * (1 - dist), feature_importances,
                                                node_relations, branch_lengths, importance_scores)



def calculate_node_importance(feature_importances, node_relations, branch_lengths, node_features):
    importance_scores = {}

    all_nodes = set(node_features.keys()) | set(node_relations.keys())
    child_nodes = set(child for children in node_relations.values() for child in children)
    root_nodes = list(all_nodes - child_nodes)

    if not root_nodes:
        raise ValueError("Cannot find root node. Make sure the node_relations are correct.")

    root_node = root_nodes[0]

    calculate_node_importance_recursive(root_node, 0, feature_importances, node_relations, branch_lengths,
                                        importance_scores)
    return importance_scores



def save_node_relations_nwk(node_relations, root_node):
    def recursive_nwk(node):
        if node not in node_relations or not node_relations[node]:
            return node
        children = node_relations[node]
        children_nwk = [recursive_nwk(child) for child in children]
        return f"({','.join(children_nwk)}){node}"

    return recursive_nwk(root_node) + ";"


def main():
    args = get_config_feature_importance()

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    node_features = {k: torch.tensor(v) for k, v in data['features'].items()}
    node_relations = data['relations']
    branch_lengths = data['branch_lengths']
    true_labels = np.array(data['true_labels'])

    node_features_normalized = normalize_features(node_features)

    feature_importances = calculate_feature_information_gain(node_features_normalized, true_labels)

    importance_scores = calculate_node_importance(feature_importances, node_relations, branch_lengths, node_features)

    importance_df = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['Importance'])

    leaf_nodes = [node for node in node_features_normalized.keys() if node not in node_relations.keys()]
    internal_nodes = [node for node in node_relations.keys()]

    leaf_df = importance_df.loc[leaf_nodes].sort_values(by='Importance', ascending=False)
    internal_df = importance_df.loc[internal_nodes].sort_values(by='Importance', ascending=False)
    all_nodes_df = importance_df.sort_values(by='Importance', ascending=False)

    def normalize_importance(df):
        total_importance = df['Importance'].sum()
        if total_importance > 0:
            df['Normalized Importance'] = df['Importance'] / total_importance
        else:
            df['Normalized Importance'] = df['Importance']
        return df[['Normalized Importance']]

    leaf_df = normalize_importance(leaf_df)
    internal_df = normalize_importance(internal_df)
    all_nodes_df = normalize_importance(all_nodes_df)


    node_relations_df = pd.DataFrame([
        {"Parent": parent, "Child": child}
        for parent, children in node_relations.items()
        for child in children
    ])

    all_nodes = set(node_features_normalized.keys()) | set(node_relations.keys())
    child_nodes = set(child for children in node_relations.values() for child in children)
    root_nodes = list(all_nodes - child_nodes)
    root_node = root_nodes[0]
    nwk_str = save_node_relations_nwk(node_relations, root_node)

    output_path = f'{args.o}PhyloSpec_Feature_Importance_Score.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        leaf_df.to_excel(writer, sheet_name='Leaf_Importance', index=True)
        internal_df.to_excel(writer, sheet_name='Int_Importance', index=True)
        all_nodes_df.to_excel(writer, sheet_name='All_Importance', index=True)
        node_relations_df.to_excel(writer, sheet_name='Node_Relations', index=False)
        pd.DataFrame({"nwk_format": [nwk_str]}).to_excel(writer, sheet_name='Node_Relations_NWK_Format', index=False)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()



