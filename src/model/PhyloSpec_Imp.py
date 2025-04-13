import sys
import torch
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
sys.path.append('./')
from src.global_config import get_config_feature_importance

# Normalize each node's feature matrix
def normalize_features(node_features):
    scaler = StandardScaler()
    normalized_features = {}
    for node, features in node_features.items():
        if features.dim() == 2:
            flattened_features = features.numpy()
        elif features.dim() > 2:
            # Flatten higher dimensional features
            flattened_features = features.view(features.size(0), -1).numpy()
        else:
            raise ValueError(f"Unexpected feature dimensions for node {node}: {features.dim()}")

        if flattened_features.size == 0:
            normalized_features[node] = torch.tensor(flattened_features, dtype=torch.float32)
            continue

        # Standardize features (mean=0, std=1)
        normalized = scaler.fit_transform(flattened_features)
        normalized_features[node] = torch.tensor(normalized, dtype=torch.float32)
    return normalized_features

# Compute entropy for a label distribution
def entropy(y):
    if len(y) == 0:
        return 0.0
    counts = Counter(y)
    probabilities = np.array(list(counts.values())) / len(y)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# Compute best information gain for a single feature
def best_information_gain(X, y, feature_idx, parent_entropy):
    sorted_indices = np.argsort(X[:, feature_idx])
    sorted_X = X[sorted_indices, feature_idx]
    sorted_y = y[sorted_indices]

    unique_values = np.unique(sorted_X)
    if len(unique_values) <= 1:
        return 0.0

    # Try thresholds between unique values
    thresholds = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

    max_info_gain = 0.0
    for thresh in thresholds:
        left_indices = sorted_X <= thresh
        right_indices = sorted_X > thresh

        y_left = sorted_y[left_indices]
        y_right = sorted_y[right_indices]

        # Skip empty splits
        if len(y_left) > 0 and len(y_right) > 0:
            weighted_entropy = (len(y_left) / len(sorted_y)) * entropy(y_left) + \
                               (len(y_right) / len(sorted_y)) * entropy(y_right)
            info_gain = parent_entropy - weighted_entropy
            max_info_gain = max(max_info_gain, info_gain)

    return max_info_gain

# Compute info gain vector for all features at one node
def calculate_feature_information_gain_parallel(node, features, y):
    X = features.numpy()
    if X.shape[0] == 0:
        return (node, np.zeros(X.shape[1]))

    parent_entropy = entropy(y)
    n_features = X.shape[1]
    gains = [best_information_gain(X, y, feature_idx, parent_entropy) for feature_idx in range(n_features)]
    return (node, np.array(gains))

# Wrapper: compute info gain for all nodes
def calculate_feature_information_gain(node_features, true_labels):
    feature_importances = {}
    for node in node_features.keys():
        result = calculate_feature_information_gain_parallel(node, node_features[node], true_labels)
        feature_importances[result[0]] = result[1]
    return feature_importances

# Recursively aggregate importance from parent to children
def calculate_node_importance_recursive(node, parent_importance, feature_importances, node_relations, branch_lengths, importance_scores):
    gains = feature_importances.get(node, np.array([]))
    node_importance = np.max(gains) if gains.size > 0 else 0.0

    # Add current node’s gain to parent contribution
    current_importance = node_importance + parent_importance
    importance_scores[node] = current_importance

    # Recursively propagate to children
    if node in node_relations:
        for child in node_relations[node]:
            dist = branch_lengths.get(child, 0.0)
            # Apply decay based on branch length
            calculate_node_importance_recursive(child, current_importance * (1 - dist),
                                                feature_importances, node_relations,
                                                branch_lengths, importance_scores)

# Entry: compute all node importance scores
def calculate_node_importance(feature_importances, node_relations, branch_lengths, node_features):
    importance_scores = {}

    all_nodes = set(node_features.keys()) | set(node_relations.keys())
    child_nodes = set(child for children in node_relations.values() for child in children)
    root_nodes = list(all_nodes - child_nodes)

    if not root_nodes:
        raise ValueError("Cannot find root node. Make sure the node_relations are correct.")

    root_node = root_nodes[0]
    calculate_node_importance_recursive(root_node, 0, feature_importances, node_relations, branch_lengths, importance_scores)

    return importance_scores

# Main procedure
def main():
    args = get_config_feature_importance()
    print("Start Calculation ... ")

    # Load data: features, relations, branch lengths, true labels
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    node_features = {k: torch.tensor(v) for k, v in data['features'].items()}
    node_relations = data['relations']
    branch_lengths = data['branch_lengths']
    true_labels = np.array(data['true_labels'])

    # Normalize node features
    node_features_normalized = normalize_features(node_features)

    # Step 1: compute info gain of each node
    feature_importances = calculate_feature_information_gain(node_features_normalized, true_labels)

    # Step 2: propagate node importance based on topology and branch lengths
    importance_scores = calculate_node_importance(feature_importances, node_relations, branch_lengths, node_features)

    # Step 3: normalize scores and save to Excel
    importance_df = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['Importance'])
    importance_df['Importance'] = 1 - importance_df['Importance']  # Invert scores: higher = more important
    importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()

    leaf_nodes = [node for node in node_features_normalized.keys() if node not in node_relations.keys()]
    importance_df = importance_df.loc[leaf_nodes]

    output_path = f'{args.o}PhyloSpec_Feature_Importance_Score.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        importance_df.loc[leaf_nodes].sort_values(by='Importance', ascending=False).to_excel(writer, sheet_name='Leaf_node_importance')

    print("Complete Calculation")
    print(f"PhyloSpec_Feature_Importance_Score.xlsx saved to {output_path}")

if __name__ == "__main__":
    main()
