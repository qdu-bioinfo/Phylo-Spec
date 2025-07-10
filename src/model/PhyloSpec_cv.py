import os
import pickle

import torch
import sys
import numpy as np
from Bio import Phylo
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from data_processing import load_and_preprocess_data, match_leaf_nodes, assign_unique_names, get_conv_order, \
    calculate_node_weights, process_unclassified_features, tree_p
from PhyloSpec import PhyloSpec, AuxiliaryModel, calculate_fc1_input_dim
from training_evaluating import calculate_roc_auc, cv_train_and_evaluate
import random
import torch.nn
sys.path.append('./')
from src.global_config import get_config_train_test


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
label_encoder = LabelEncoder()


"""Cross-validation training and evaluation pipeline."""
def cv_function(config, seed):
    set_seed(seed)

    csv_path = config.c
    newick_path = config.t
    taxonomy_path = config.taxo

    tree = tree_p(csv_path, newick_path)
    tree = Phylo.read(tree, 'newick')
    tree = assign_unique_names(tree)

    X, y, encoder, data = load_and_preprocess_data(csv_path, tree)
    leaf_to_species = match_leaf_nodes(tree, data)
    nodes, parents, conv_order, node_relations = get_conv_order(tree)

    if any('Unclassified' in col or 'unclassified' in col for col in data.columns):
        data, tree = process_unclassified_features(tree, data, taxonomy_path)
        X = data.iloc[:, 1:-1].values
        y = label_encoder.fit_transform(data.iloc[:, -1].values)

    num_classes = len(np.unique(y))
    node_weights = calculate_node_weights(tree)

    fold_auc = []

    if config.pkl and os.path.exists(config.pkl):
        with open(config.pkl, 'rb') as f:
            skf_splits = pickle.load(f)
        print("Using predefined fold indices from:", config.pkl)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        skf_splits = list(skf.split(X, y))
        print("No pkl provided, generating fold indices with random.")

    for fold_idx, (train_idx, val_idx) in enumerate(skf_splits):
        set_seed(seed)

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=seed)
        X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

        scaler = StandardScaler()
        X_train_smote = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        X_train_tensor = torch.tensor(X_train_smote, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)

        if num_classes == 2:
            y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1)
            y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
                                                   batch_size=6, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor),
                                                 batch_size=64, shuffle=False)

        aux_model = AuxiliaryModel()
        fc1_input_dim = calculate_fc1_input_dim(aux_model, X_train_smote, conv_order, data, leaf_to_species,
                                                node_weights)

        if num_classes == 2:
            model = PhyloSpec(fc1_input_dim=fc1_input_dim, num_res_blocks=1, channel=config.ch,
                              kernel_size=config.ks, out_feature=1).to('cpu')
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            model = PhyloSpec(fc1_input_dim=fc1_input_dim, num_res_blocks=1, channel=config.ch,
                              kernel_size=config.ks, out_feature=num_classes).to('cpu')
            criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0001)

        best_model, test_group, all_preds = cv_train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, conv_order, data, leaf_to_species, node_weights,
            num_epochs=config.ep, num_classes=num_classes
        )

        y_val_encoded = np.array(test_group)
        y_score = np.array(all_preds)

        if num_classes == 2:
            roc_auc = [calculate_roc_auc(y_val_encoded, y_score, num_classes)]
        else:
            roc_auc = calculate_roc_auc(y_val_encoded, y_score, num_classes)

        fold_auc.append(roc_auc)

    fold_auc = np.array(fold_auc)  # shape: (n_folds, n_classes)
    average_auc_per_class = np.mean(fold_auc, axis=0)

    print("\nFinal Average ROC AUC per Class:")
    for i, auc in enumerate(average_auc_per_class):
        print(f"Class {i} AUC: {auc:.4f}")


"""Main entry point."""
def main():
    seed = 42
    config = get_config_train_test()
    if config.PhyloSpec == 'cv':
        cv_function(config, seed)
    else:
        print("Invalid mode. Use 'cv'.")
        return

if __name__ == '__main__':
    main()
