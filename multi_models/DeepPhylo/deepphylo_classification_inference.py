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


def test(X_test,Y_test,phy_embedding):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    kernel_size_conv = args.kernel_size_conv
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
    test_dataset = DeepPhyDataset(phy_embedding, X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.custom_collate_fn)
    model = DeepPhylo_classification(hidden_size, test_dataset.embeddings, kernel_size_conv, kernel_size_pool, activation=activation).to(device)
    metrics_dict = {'acc': [], 'mcc': [], 'roc_auc': [], 'aupr': [], 'precision':[], 'recall':[],'specificity':[], 'sensitivity':[]}
    model.load_state_dict(torch.load("data/gender_classification/best_model.pth"))
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for batch in test_loader:
            test_true.append(batch['y'].numpy())
            batch = {key: val.to(device) for key, val in batch.items()}
            y_pred_test =model(batch['X'], batch['nonzero_indices'])
            test_preds.append(y_pred_test.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_preds = np.concatenate(test_preds)
    metrics = compute_metrics(test_true,test_preds)
    metrics_dict['acc'].append(metrics['acc'])
    metrics_dict['mcc'].append(metrics['mcc'])
    metrics_dict['roc_auc'].append(metrics['roc_auc'])
    metrics_dict['aupr'].append(metrics['aupr'])
    metrics_dict['precision'].append(metrics['precision'])
    metrics_dict['recall'].append(metrics['recall'])
    metrics_dict['specificity'].append(metrics['specificity'])
    metrics_dict['sensitivity'].append(metrics['sensitivity'])
    
    return metrics_dict, test_preds




if __name__ == '__main__':
    """
    #python deepphylo_classification_inference.py 
    # -test_X 'data/gender_classification/X_test.npy' 
    # -test_Y 'data/gender_classification/Y_test.npy'  
    # -hs 80 -kec 3 -l 0.0001 -bs 32 -kep 7 -act relu 

    """
    set_seed(1234)
    parser = argparse.ArgumentParser(
        description='Command line tool for twin classification')
    parser.add_argument('-test_X',
                        '--test_X',
                        default='data/gender_classification/X_test.npy',
                        type=str,
                        help='Path to the test data')
    parser.add_argument('-test_Y',
                        '--test_Y',
                        default=None,
                        type=str,
                        help='Path to the true data')
    parser.add_argument('-hs',
                        '--hidden_size',
                        default=80,
                        type=int,
                        help='Hidden_size which using in pca dimensionality reduction operation')
    parser.add_argument('-kec',
                        '--kernel_size_conv',
                        default=3,
                        type=int,
                        help='Kernal_size which applied to convolutional layers')
    parser.add_argument('-l',
                        '--lr',
                        default=1e-4,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('-bs',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='Batchsize size when encoding protein embedding with backbone')
    parser.add_argument('-kep',
                        '--kernel_size_pool',
                        default=7,
                        type=int,
                        help='Kernal_size which applied to pooling layers')
    parser.add_argument('-act',
                        '--activation',
                        default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function for encoding protein embedding with backbone (default: relu)')

    args = parser.parse_args()
    X_test = np.load(args.test_X)
    Y_test = np.load(args.test_Y) if args.test_Y is not None else args.test_Y

    # X_test = np.load('data/gender_classification/X_test.npy')
    # Y_test = np.load('data/gender_classification/Y_test.npy')
    phy_embedding = np.load('data/gender_classification/phy_embedding_gender.npy')
    if Y_test is not None and Y_test.any():
        metrics_dict,test_preds = test(X_test, Y_test, phy_embedding)
        recall, precision = plot_pr_curve(metrics_dict['precision'][0], metrics_dict['recall'][0])
        specificity, sensitivity = plot_ss_curve(metrics_dict['sensitivity'][0], metrics_dict['specificity'][0])
        print(f"Best metrics: sens: {sensitivity:.4f}, spec:{specificity:.4f}, acc: {metrics_dict['acc'][0]:.4f}, p:{precision:.4f},mcc: {metrics_dict['mcc'][0]:.4f}, f1:{2*precision*recall / (precision+recall):.4f},roc_auc: {metrics_dict['roc_auc'][0]:.4f}, aupr: {metrics_dict['aupr'][0]:.4f}")
    else:
        Y_test = np.random.randint(0, 2, size=X_test.shape[0])
        metrics_dict,test_preds = test(X_test, Y_test, phy_embedding)
        print(f"test_preds:",test_preds)