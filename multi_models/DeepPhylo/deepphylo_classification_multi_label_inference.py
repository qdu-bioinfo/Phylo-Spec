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
from deepphylo.evaluate import compute_metrics_multi_label, select_best_epoch
from deepphylo.model import DeepPhylo_multi_label
import argparse
import warnings
warnings.filterwarnings("ignore")   

def test(X_test,Y_test,phy_embedding):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.custom_collate_fn)
    model = DeepPhylo_multi_label(args.hidden_size, test_dataset.embeddings, args.kernel_size_conv, args.kernel_size_pool, activation=activation).to(device)
    metrics_dict = {'acc': [], 'mcc': [], 'roc_auc': [], 'aupr': [], 'f1': []}
    model.load_state_dict(torch.load("data/ggmp_multi_label_classification/best_model.pth"))
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
    _, metrics_all = compute_metrics_multi_label(test_true,test_preds)
    metrics_dict['acc'].append(metrics_all['acc'])
    metrics_dict['mcc'].append(metrics_all['mcc'])
    metrics_dict['roc_auc'].append(metrics_all['roc_auc'])
    metrics_dict['aupr'].append(metrics_all['aupr'])
    metrics_dict['f1'].append(metrics_all['f1'])
    
    return metrics_dict, test_preds




if __name__ == '__main__':
    """
        # python3 deepphylo_classification_multi_label_inference.py --test_X data/ggmp_multi_label_classification/X_test.npy 
        # --test_Y data/ggmp_multi_label_classification/y_test_mets_gastritis_t2dm_gout.npy 
        # --hs 500 --kec 7 --l 1e-4 --bs 64 --kep 4 --act relu

    """
    set_seed(1234)
    parser = argparse.ArgumentParser(
        description='Command line tool for multi_label classification')
    parser.add_argument('-test_X',
                        '--test_X',
                        default='data/ggmp_multi_label_classification/X_test.npy',
                        type=str,
                        help='Path to the test data')
    parser.add_argument('-test_Y',
                        '--test_Y',
                        default='data/ggmp_multi_label_classification/y_test_mets_gastritis_t2dm_gout.npy',
                        type=str,
                        help='Path to the true data')
    parser.add_argument('-hs',
                        '--hidden_size',
                        default=32,
                        type=int,
                        help='Hidden_size which using in pca dimensionality reduction operation')
    parser.add_argument('-kec',
                        '--kernel_size_conv',
                        default=5,
                        type=int,
                        help='Kernal_size which applied to convolutional layers')
    parser.add_argument('-l',
                        '--lr',
                        default=5e-2,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('-bs',
                        '--batch_size',
                        default=8,
                        type=int,
                        help='Batchsize size when encoding protein embedding with backbone')
    parser.add_argument('-kep',
                        '--kernel_size_pool',
                        default=1,
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

    C = np.load('data/ggmp_multi_label_classification/distance_matrix.npy')
    D = inverse_C(C)
    phy_embedding = reducer(C, 'pca', args.hidden_size, whiten=True)
    if Y_test is not None and Y_test.any():
        metrics_dict,test_preds = test(X_test, Y_test, phy_embedding)
        print(f"Best metrics: acc: {metrics_dict['acc'][0]:.4f}, mcc: {metrics_dict['mcc'][0]:.4f}, f1:{metrics_dict['f1'][0]:.4f}, roc_auc: {metrics_dict['roc_auc'][0]:.4f}, aupr: {metrics_dict['aupr'][0]:.4f}")
    else:
        Y_test = np.random.randint(0, 2, size=(X_test.shape[0],4))
        metrics_dict,test_preds = test(X_test, Y_test, phy_embedding)
        print(f"test_preds:",test_preds)