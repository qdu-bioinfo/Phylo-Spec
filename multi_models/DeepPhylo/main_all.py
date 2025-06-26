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


def plot_training(train_losses, val_losses):
    # plot the training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(frameon=False)
    plt.show()


def train(X_train, Y_train, X_eval, Y_eval, phy_embedding, batch_size=32, lr=1e-4, hidden_size=32, kernal_size_conv=13, kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCELoss()
    batch_size = batch_size
    # Create DataLoader for training and validation data
    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.custom_collate_fn)
    val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.custom_collate_fn)
    model = DeepPhylo(hidden_size=hidden_size,
                    embeddings=train_dataset.embeddings,
                    kernel_size_conv=kernal_size_conv,
                    kernel_size_pool=kernel_size_pool,
                    dropout_conv=dropout_conv,
                    activation=activation).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Training
    epochs = 200
    patience = 20
    best_val_loss = float("inf")
    counter = 0
    train_losses = []
    val_losses = []
    val_pred_labels = []
    val_true_labels = []
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            y_pred_train = model(batch['X'], batch['nonzero_indices'])
            y_pred_train = y_pred_train.squeeze(dim=1)
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
                y_pred_val = model(batch['X'], batch['nonzero_indices'])
                y_pred_val = y_pred_val.squeeze(dim=1)
                loss_val = criterion(y_pred_val, batch['y'])
                val_loss += loss_val.item() * batch['X'].size(0)
                val_preds.append(y_pred_val.detach().cpu().numpy())
        val_loss /= len(val_loader.dataset)
        # Calculate validation R2
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
    # 获取otu数量
    otu_num = phy_embedding.shape[0]

    # 创建索引数组
    indices = np.arange(otu_num)

    # 随机选择一部分索引并打乱
    if portion == 0:
        return phy_embedding, X_train, X_eval
    else:
        num_to_shuffle = int(portion * otu_num)
        shuffle_indices = np.random.choice(indices, size=num_to_shuffle, replace=False)
        np.random.shuffle(shuffle_indices)

        # 用打乱的索引替换原来的索引
        indices[:num_to_shuffle] = shuffle_indices

        # 用新的索引数组重新排列phy_embedding和X_train
        phy_embedding = phy_embedding[indices]
        X_train = X_train[:, indices]
        X_eval = X_eval[:, indices]
        return phy_embedding, X_train, X_eval


if __name__ == '__main__':
    set_seed(1234)

    parser = argparse.ArgumentParser(
        description='Command line tool for IBD diagnosis')

    parser.add_argument('-hs',
                        '--hidden_size',
                        default=16,
                        type=int,
                        help='Hidden_size which using in pca dimensionality reduction operation')
    parser.add_argument('-kec',
                        '--kernal_size_conv',
                        default=7,
                        type=int,
                        help='Kernal size which applied to convolutional layers')
    parser.add_argument('-kep',
                        '--kernal_size_pool',
                        default=4,
                        type=int,
                        help='pooling size of convolutional layers')
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
    parser.add_argument('-act',
                        '--activation',
                        default='tanh',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function for encoding protein embedding with backbone (default: relu)')
    parser.add_argument('-d',
                        '--dropout',
                        default=0.5,
                        type=float,
                        help='dropout rate of convolutional layers')
    parser.add_argument('-p',
                    '--portion_shuffle',
                    default=0.0,
                    type=float,
                    help='proportion of feature embeddings to be shuffled')
    args = parser.parse_args()

    # Load data
    X = np.load(r'C:\Users\91566\Desktop\X_converted.npy')
    y = np.load(r'C:\Users\91566\Desktop\y_converted.npy')
    phy_embedding = np.load(r'C:\Users\91566\Desktop\embedding.npy')

    # Generate all combinations of hyperparameters
    hidden_size = args.hidden_size
    kernal_size_conv = args.kernal_size_conv
    kernel_size_pool = args.kernal_size_pool
    dropout_conv = args.dropout
    portion = args.portion_shuffle

    if args.activation == 'relu':
        activation = nn.ReLU()
    elif args.activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif args.activation == 'tanh':
        activation = nn.Tanh()
    else:
        raise ValueError("Invalid activation function")
    lr = args.lr
    batch_size = args.batchsize

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    all_preds = []
    all_true = []

    for train_index, val_index in kf.split(X):
        X_train, X_eval = X[train_index], X[val_index]
        Y_train, Y_eval = y[train_index], y[val_index]

        phy_embedding, X_train, X_eval = random_shuffle(phy_embedding, X_train, X_eval, portion=portion)

        print(f'Number of validation samples: {len(Y_eval)}, +: {sum(Y_eval)}, -: {len(Y_eval) - sum(Y_eval)}')
        train_losses, val_losses, val_true_labels, val_pred_labels = train(X_train, Y_train, X_eval, Y_eval, phy_embedding, batch_size=batch_size, lr=lr, hidden_size=hidden_size, kernal_size_conv=kernal_size_conv, kernel_size_pool=kernel_size_pool, dropout_conv=dropout_conv, activation=activation)

        best_epoch = select_best_epoch(val_losses)
        # 保存最佳epoch的预测
        all_true.append(val_true_labels[best_epoch])
        all_preds.append(val_pred_labels[best_epoch])

    # 计算AUC
    all_true = np.concatenate(all_true)
    all_preds = np.concatenate(all_preds)
    auc = roc_auc_score(all_true, all_preds)
    print(f'AUC: {auc}')
