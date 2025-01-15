import os
import sys
import torch
import numpy as np
import joblib
from Bio import Phylo
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from data_processing import load_and_preprocess_data, match_leaf_nodes, assign_unique_names, get_conv_order, \
    calculate_node_weights, save_node_features_with_pickle, process_unclassified_features
from PhyloSpec import PhyloSpec, AuxiliaryModel, calculate_fc1_input_dim
sys.path.append('./')
from src.global_config import get_config_train_test
from training_evaluating import train_model, evaluate_model_on_test

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
label_encoder = LabelEncoder()

def train_model_function(config, seed):

    newick_path = config.t
    train_csv_path = config.c
    taxonomy_path = config.taxo
    tree = Phylo.read(newick_path, 'newick')
    tree = assign_unique_names(tree)



    X_train, y_train, encoder, data_train = load_and_preprocess_data(train_csv_path, tree)

    if any("Unclassified" in col or "unclassified" in col for col in data_train.columns):
        data_train, tree = process_unclassified_features(tree, data_train, taxonomy_path)
        X_train = data_train.iloc[:, 1:-1].values
        # y_train = data_train.iloc[:, -1].values
        y_train = label_encoder.fit_transform(data_train.iloc[:, -1].values)

    leaf_to_species = match_leaf_nodes(tree, data_train)
    nodes, parents, conv_order, node_relations = get_conv_order(tree)

    node_weights = calculate_node_weights(tree)

    smote = SMOTE(random_state=seed)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.bs, shuffle=True)

    aux_model = AuxiliaryModel(channel=config.ch, kernel_size=config.ks)
    fc1_input_dim = calculate_fc1_input_dim(aux_model, X_train, conv_order, data_train, leaf_to_species,
                                            node_weights)

    final_model = PhyloSpec(fc1_input_dim=fc1_input_dim, num_res_blocks=1, channel=config.ch,
                      kernel_size=config.ks).to('cpu')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=config.lr, weight_decay=0.0001)


    print("Training the model on the entire training set...")
    final_model = train_model(
        final_model, train_loader, criterion, optimizer, conv_order, data_train, leaf_to_species,
        node_weights=node_weights, num_epochs=config.ep
    )

    all_train_labels = []
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        all_train_labels.append(batch_labels)

    all_train_labels = torch.cat(all_train_labels, dim=0).numpy()

    node_features = final_model.accumulated_node_features
    save_node_features_with_pickle(
        node_features,
        node_relations,
        node_weights,
        all_train_labels,
        os.path.join(config.o, 'Node_Features.pkl')
    )


    scaler_path = os.path.join(config.o, 'StandardScaler.pkl')
    joblib.dump(scaler, scaler_path)

    torch.save(final_model, config.o+'train_model.pth')

    print("End of training")




def test_model_function(config, seed):
    set_seed(seed)

    newick_path = config.t
    test_csv_path = config.c
    taxonomy_path = config.taxo
    tree = Phylo.read(newick_path, 'newick')
    tree = assign_unique_names(tree)

    X_test, y_test, _, data_test = load_and_preprocess_data(test_csv_path, tree)
    if any("Unclassified" in col or "unclassified" in col for col in data_test.columns):
        data_test, tree = process_unclassified_features(tree, data_test, taxonomy_path)
        X_test = data_test.iloc[:, 1:-1].values
        # y_test = data_test.iloc[:, -1].values
        y_test = label_encoder.fit_transform(data_test.iloc[:, -1].values)

    leaf_to_species = match_leaf_nodes(tree, data_test)
    nodes, parents, conv_order, node_relations = get_conv_order(tree)

    node_weights = calculate_node_weights(tree)

    scaler = joblib.load(config.o+'StandardScaler.pkl')

    X_test = scaler.transform(X_test)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.bs, shuffle=False)

    final_model = torch.load(config.o+'train_model.pth')

    print("Testing the model on the test set...")
    y_true, y_scores = evaluate_model_on_test(final_model, test_loader, conv_order, data_test, leaf_to_species,
                                              node_weights)

    roc_auc = roc_auc_score(y_true, y_scores)
    print(f"ROC AUC: {roc_auc:.4f}")

def main():
    config = get_config_train_test()

    seed = 42
    set_seed(seed)

    if config.PhyloSpec == 'train':
        train_model_function(config, seed)
    elif config.PhyloSpec == 'test':
        test_model_function(config, seed)
    else:
        print("Invalid mode. Use 'train' or 'test'.")
        return

if __name__ == '__main__':
    main()