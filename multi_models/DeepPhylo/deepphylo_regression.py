import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import r2_score
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from deepphylo.pre_dataset import set_seed,reducer, inverse_C, DeepPhyDataset
from deepphylo.model import DeepPhylo_regression
import argparse


def train(X_train, Y_train, X_eval, Y_eval, phy_embedding):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    kernal_size_conv = args.kernal_size_conv
    num_layers = 1
    criterion = nn.SmoothL1Loss(beta=0.8)
    batch_size = args.batchsize
    kernal_size_pool = args.kernal_size_pool
    if args.activation == 'relu':
        activation = nn.ReLU()
    elif args.activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif args.activation == 'tanh':
        activation = nn.Tanh()
    else:
        raise ValueError("Invalid activation function")
    # Create DataLoader for training and validation data
    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.custom_collate_fn)
    val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.custom_collate_fn)
    model = DeepPhylo_regression(hidden_size, train_dataset.embeddings,kernal_size_conv, kernal_size_pool, activation=activation).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Training
    epochs = args.epochs
    patience = 5
    best_val_loss = float("inf")
    counter = 0
    train_losses = []
    val_losses = []
    val_r2s = []
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            y_pred_train = model(batch['X'], batch['nonzero_indices'])
            loss_train = criterion(y_pred_train, batch['y'])
            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item() * batch['X'].size(0)

        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        y_val = []
        with torch.no_grad():
            for batch in val_loader:
                y_val.append(batch['y'].numpy())
                batch = {key: val.to(device) for key, val in batch.items()}
                y_pred_val =model(batch['X'], batch['nonzero_indices'])
                loss_val = criterion(y_pred_val, batch['y'])
                val_loss += loss_val.item() * batch['X'].size(0)
                val_preds.append(y_pred_val.detach().cpu().numpy())
        val_loss /= len(val_loader.dataset)
        # Calculate validation R2
        y_val = np.concatenate(y_val)
        val_preds = np.concatenate(val_preds)
        val_r2 = r2_score(y_val, val_preds)
        print(f'epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_r2: {val_r2:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2s.append(val_r2)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    return train_losses, val_losses, val_r2s



if __name__ == '__main__':
    set_seed(1234)
    parser = argparse.ArgumentParser(
        description='Command line tool for twin classification')

    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-hs',
                        '--hidden_size',
                        default=64,
                        type=int,
                        help='Hidden_size which using in pca dimensionality reduction operation')
    parser.add_argument('-kec',
                        '--kernal_size_conv',
                        default=7,
                        type=int,
                        help='Kerna;_size which applied to convolutional layers')
    parser.add_argument('-l',
                        '--lr',
                        default=1e-4,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('-bs',
                        '--batchsize',
                        default=64,
                        type=int,
                        help='Batchsize size when encoding protein embedding with backbone')
    parser.add_argument('-kep',
                        '--kernal_size_pool',
                        default=4,
                        type=int,
                        help='Kernal_size which applied to pooling layers')
    parser.add_argument('-act',
                        '--activation',
                        default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function for encoding protein embedding with backbone (default: relu)')
    args = parser.parse_args()
    X_train = np.load('data/age_regression/X_train.npy')
    X_eval = np.load('data/age_regression/X_eval.npy')
    Y_train = np.load('data/age_regression/Y_train.npy')
    Y_eval = np.load('data/age_regression/Y_eval.npy')
    C = np.load('data/age_regression/c.npy')
    D = inverse_C(C)
    phy_embedding = reducer(C, 'pca', args.hidden_size, whiten=True)

    # hac_index = hac(D)
    # phy_embedding = phy_embedding[hac_index,:]
    # X_train = X_train[:,hac_index]
    # X_eval = X_eval[:,hac_index]

    train_losses, val_losses, val_r2s = train(X_train, Y_train, X_eval, Y_eval, phy_embedding)
    #plot_age(train_losses, val_losses, val_r2s, title='The DeepPhy model prediction on age dataset: train and test Loss/R2')


    # 绘制长度分布图
    # len_train = [len(np.nonzero(X_train[idx])[0]) for idx in range(len(X_train))]
    # plt.bar([i for i in range(len(len_train))], sorted(len_train))