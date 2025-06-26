import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from deepphylo.pre_dataset import set_seed,reducer, inverse_C, DeepPhyDataset
from deepphylo.plot import plot_ss_curve, plot_pr_curve
from deepphylo.evaluate import compute_metrics, select_best_epoch
from deepphylo.model import DeepPhylo_classification
import argparse


def train(X_train, Y_train, X_eval, Y_eval, phy_embedding):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    kernel_size_conv = args.kernel_size_conv
    criterion = nn.MSELoss()
    batch_size = args.batch_size
    kernel_size_pool = args.kernel_size_pool
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
    model = DeepPhylo_classification(hidden_size, train_dataset.embeddings, kernel_size_conv, kernel_size_pool, activation=activation).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Training
    epochs = args.epochs
    patience = 5
    best_val_loss = float("inf")
    counter = 0
    train_losses = []
    val_losses = []
    metrics_dict = {'acc': [], 'mcc': [], 'roc_auc': [], 'aupr': [], 'precision':[], 'recall':[],'specificity':[], 'sensitivity':[]}
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
        metrics = compute_metrics(y_val, val_preds)
        print(f"epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, acc: {metrics['acc']:.4f}, mcc:{metrics['mcc']:.4}, roc-auc:{metrics['roc_auc']:.4f}, aupr:{metrics['aupr']:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_dict['acc'].append(metrics['acc'])
        metrics_dict['mcc'].append(metrics['mcc'])
        metrics_dict['roc_auc'].append(metrics['roc_auc'])
        metrics_dict['aupr'].append(metrics['aupr'])
        metrics_dict['precision'].append(metrics['precision'])
        metrics_dict['recall'].append(metrics['recall'])
        metrics_dict['specificity'].append(metrics['specificity'])
        metrics_dict['sensitivity'].append(metrics['sensitivity'])

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
    return train_losses, val_losses, metrics_dict




if __name__ == '__main__':
    """
    #python deepphylo_classification.py --epochs 500 -hs 80 -kec 3 -l 0.0001 -bs 32 -kep 7 -act relu       

    """
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
                        '--kernel_size_conv',
                        default=7,
                        type=int,
                        help='Kernal_size which applied to convolutional layers')
    parser.add_argument('-l',
                        '--lr',
                        default=1e-4,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('-bs',
                        '--batch_size',
                        default=64,
                        type=int,
                        help='Batchsize size when encoding protein embedding with backbone')
    parser.add_argument('-kep',
                        '--kernel_size_pool',
                        default=4,
                        type=int,
                        help='Kernal_size which applied to pooling layers')
    parser.add_argument('-act',
                        '--activation',
                        default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function for encoding protein embedding with backbone (default: relu)')


    args = parser.parse_args()
    X_train = np.load('data/gender_classification/X_train.npy')
    X_eval = np.load('data/gender_classification/X_eval.npy')
    Y_train = np.load('data/gender_classification/Y_train.npy')
    Y_eval = np.load('data/gender_classification/Y_eval.npy')
    C = np.load('data/gender_classification/c.npy')
    D = inverse_C(C)
    phy_embedding = reducer(C, 'pca', args.hidden_size, whiten=True)
    train_losses, val_losses, metrics_dict = train(X_train, Y_train, X_eval, Y_eval, phy_embedding)
    best_epoch = select_best_epoch(metrics_dict, ['acc', 'mcc', 'roc_auc', 'aupr'])
    recall, precision = plot_pr_curve(metrics_dict['precision'][best_epoch], metrics_dict['recall'][best_epoch])
    specificity, sensitivity = plot_ss_curve(metrics_dict['sensitivity'][best_epoch], metrics_dict['specificity'][best_epoch])
    print(f"Best epoch:{best_epoch+1}")
    print(f"Best metrics: sens: {sensitivity:.4f}, spec:{specificity:.4f}, acc: {metrics_dict['acc'][best_epoch]:.4f}, p:{precision:.4f},mcc: {metrics_dict['mcc'][best_epoch]:.4f}, f1:{2*precision*recall / (precision+recall):.4f},roc_auc: {metrics_dict['roc_auc'][best_epoch]:.4f}, aupr: {metrics_dict['aupr'][best_epoch]:.4f}")