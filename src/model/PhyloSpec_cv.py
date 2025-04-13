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
    calculate_node_weights, process_unclassified_features
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


def cv_function(config, seed):
    seed = seed
    set_seed(seed)

    csv_path = config.c
    newick_path = config.t
    taxonomy_path = config.taxo

    tree = Phylo.read(newick_path, 'newick')
    tree = assign_unique_names(tree)

    X, y, encoder, data = load_and_preprocess_data(csv_path, tree)
    leaf_to_species = match_leaf_nodes(tree, data)
    nodes, parents, conv_order, node_relations = get_conv_order(tree)

    if any('Unclassified' in col or 'unclassified' in col for col in data.columns):
        data, tree = process_unclassified_features(tree, data, taxonomy_path)
        X = data.iloc[:, 1:-1].values
        y = label_encoder.fit_transform(data.iloc[:, -1].values)

    num_classes = len(np.unique(y))  # Number of classes (2 for binary, >2 for multiclass)

    node_weights = calculate_node_weights(tree)

    fold_auc = []

    if config.pkl and os.path.exists(config.pkl):  # 如果提供了pkl文件
        with open(config.pkl, 'rb') as f:
            fold = pickle.load(f)  # 应该是一个 list，格式如：[(train_idx, val_idx), ...]
        print("Using predefined fold indices from:", config.pkl)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold = list(skf.split(X, y))  # 动态生成
        print("No pkl provided, generating fold indices with random seed.")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        set_seed(seed)

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        smote = SMOTE(random_state=seed)
        X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

        scaler = StandardScaler()
        X_train_smote = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

        X_train_tensor = torch.tensor(X_train_smote, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)  # Ensure labels are long for CrossEntropyLoss
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
                                                   batch_size=config.bs, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor),
                                                 batch_size=config.bs, shuffle=False)

        aux_model = AuxiliaryModel()
        fc1_input_dim = calculate_fc1_input_dim(aux_model, X_train_smote, conv_order, data, leaf_to_species,
                                                node_weights)



        # Use CrossEntropyLoss for both binary and multi-class classification
        if num_classes == 2:
            model = PhyloSpec(fc1_input_dim=fc1_input_dim, num_res_blocks=1, channel=config.ch,
                              kernel_size=config.ks, out_feature=1).to('cpu')
            criterion = torch.nn.BCEWithLogitsLoss()  # For binary classification
        else:
            model = PhyloSpec(fc1_input_dim=fc1_input_dim, num_res_blocks=1, channel=config.ch,
                              kernel_size=config.ks, out_feature=num_classes).to('cpu')
            criterion = torch.nn.CrossEntropyLoss()  # For multi-class classification

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0001)

        best_model, test_group, all_preds = cv_train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, conv_order, data, leaf_to_species, node_weights,
            num_epochs=config.ep, num_classes=num_classes
        )


        # Calculate AUC
        y_val_encoded = np.array(test_group)
        y_score = np.array(all_preds)

        # For binary classification, we need to compute AUC for the positive class (second column)
        if num_classes == 2:
            roc_auc = calculate_roc_auc(y_val_encoded, y_score, num_classes)
            # print(f"Fold {fold + 1} ROC AUC (Binary): {roc_auc:.4f}")
        else:  # For multi-class classification, compute AUC for each class
            roc_auc = calculate_roc_auc(y_val_encoded, y_score, num_classes)
            # for i in range(num_classes):
            #     print(f"Fold {fold + 1} Class {i} ROC AUC (Multiclass): {roc_auc[i]:.4f}")

        fold_auc.append(roc_auc)

    # Calculate average AUC
    average_auc = np.mean(fold_auc)
    print(f"Average ROC AUC: {average_auc:.4f}")


def main():
    seed = 42
    config = get_config_train_test()
    cv_function(config,seed)
    if config.PhyloSpec == 'cv':
        cv_function(config, seed)
    else:
        print("Invalid mode. Use 'cv'.")
        return

if __name__ == '__main__':
    main()