import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from deepphylo.utils import set_seed
from deepphylo.evaluate import compute_metrics_ibd as compute_metrics
from deepphylo.model import DeepPhylo_ibd as DeepPhylo
from deepphylo.pre_dataset import DeepPhyDataset

def train(X_train, Y_train, X_eval, Y_eval, phy_embedding,
          train_batch_size=64, val_batch_size=64,
          lr=1e-4, hidden_size=32, kernal_size_conv=13,
          kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCELoss()

    # Create DataLoader for training and validation data
    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.custom_collate_fn
    )

    val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=train_dataset.custom_collate_fn
    )

    model = DeepPhylo(
        hidden_size=hidden_size,
        embeddings=train_dataset.embeddings,
        kernel_size_conv=kernal_size_conv,
        kernel_size_pool=kernel_size_pool,
        dropout_conv=dropout_conv,
        activation=activation
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    epochs = 10
    patience = 1
    best_val_loss = float("inf")
    counter = 0
    train_losses = []
    val_losses = []
    val_pred_labels = []
    val_true_labels = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            y_pred_train = model(batch['X'], batch['nonzero_indices']).squeeze(dim=1)
            loss_train = criterion(y_pred_train, batch['y'])
            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item() * batch['X'].size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        y_val = []
        with torch.no_grad():
            for batch in val_loader:
                y_val.append(batch['y'].numpy())
                batch = {key: val.to(device) for key, val in batch.items()}
                y_pred_val = model(batch['X'], batch['nonzero_indices']).squeeze(dim=1)
                loss_val = criterion(y_pred_val, batch['y'])
                val_loss += loss_val.item() * batch['X'].size(0)
                val_preds.append(y_pred_val.detach().cpu().numpy())
        val_loss /= len(val_loader.dataset)

        # Metrics and logging
        y_val = np.concatenate(y_val)
        val_preds = np.concatenate(val_preds)
        val_pred_labels.append(val_preds)
        val_true_labels.append(y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    del model, optimizer
    return train_losses, val_losses, val_true_labels, val_pred_labels


def select_best_epoch(val_losses):
    best_epoch = np.argmin(val_losses)
    return best_epoch


def random_shuffle(phy_embedding, X_train, X_eval, portion=0.5):
    otu_num = phy_embedding.shape[0]

    indices = np.arange(otu_num)

    if portion == 0:
        return phy_embedding, X_train, X_eval
    else:
        num_to_shuffle = int(portion * otu_num)
        shuffle_indices = np.random.choice(indices, size=num_to_shuffle, replace=False)
        np.random.shuffle(shuffle_indices)
        indices[:num_to_shuffle] = shuffle_indices
        phy_embedding = phy_embedding[indices]
        X_train = X_train[:, indices]
        X_eval = X_eval[:, indices]
        return phy_embedding, X_train, X_eval

if __name__ == '__main__':
    set_seed(1234)

    parser = argparse.ArgumentParser(description='Minimal CLI for IBD diagnosis')

    parser.add_argument('-xnpy', required=True, help='Path to input X .npy file')
    parser.add_argument('-ynpy', required=True, help='Path to input y .npy file')
    parser.add_argument('-embed', required=True, help='Path to input embedding .npy file')

    args = parser.parse_args()

    X = np.load(args.xnpy)
    y = np.load(args.ynpy)
    phy_embedding = np.load(args.embed)

    hidden_size = 32
    kernal_size_conv = 13
    kernel_size_pool = 4
    dropout_conv = 0.2
    lr = 1e-4
    portion = 0.0
    activation = nn.LeakyReLU()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = []
    all_true = []

    for train_index, val_index in kf.split(X):
        X_train, X_eval = X[train_index], X[val_index]
        Y_train, Y_eval = y[train_index], y[val_index]

        phy_embedding, X_train, X_eval = random_shuffle(phy_embedding, X_train, X_eval, portion=portion)

        print(f'Number of validation samples: {len(Y_eval)}, +: {sum(Y_eval)}, -: {len(Y_eval) - sum(Y_eval)}')
        train_losses, val_losses, val_true_labels, val_pred_labels = train(
            X_train, Y_train, X_eval, Y_eval, phy_embedding,
            train_batch_size=64, val_batch_size=64,
            lr=lr, hidden_size=hidden_size,
            kernal_size_conv=kernal_size_conv, kernel_size_pool=kernel_size_pool,
            dropout_conv=dropout_conv, activation=activation)

        best_epoch = select_best_epoch(val_losses)
        all_true.append(val_true_labels[best_epoch])
        all_preds.append(val_pred_labels[best_epoch])
    all_true = np.concatenate(all_true)
    all_preds = np.concatenate(all_preds)
    auc = roc_auc_score(all_true, all_preds)
    print(f'AUC: {auc}')
