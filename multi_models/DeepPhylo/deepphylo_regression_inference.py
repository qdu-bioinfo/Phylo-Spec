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



def test(X_test,Y_test,phy_embedding):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_size = args.hidden_size
    kernal_size_conv = args.kernal_size_conv
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
    test_dataset = DeepPhyDataset(phy_embedding, X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.custom_collate_fn)
    model = DeepPhylo_regression(hidden_size, test_dataset.embeddings,kernal_size_conv, kernal_size_pool, activation=activation).to(device)
    model.load_state_dict(torch.load("data/age_regression/best_model.pth"))
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for batch in test_loader:
            test_true.append(batch['y'].numpy())
            batch = {key: val.to(device) for key, val in batch.items()}
            y_pred_test =model(batch['X'], batch['nonzero_indices'])
            test_preds.append(y_pred_test.detach().cpu().numpy())
    # Calculate validation R2
    test_true = np.concatenate(test_true)
    test_preds = np.concatenate(test_preds)
    test_r2 = r2_score(test_true, test_preds)
    # print(f' test_r2: {test_r2:.4f}')

    return test_preds,test_r2



if __name__ == '__main__':
    set_seed(1234)
    parser = argparse.ArgumentParser(
        description='Command line tool for twin classification')
    parser.add_argument('-test_X',
                        '--test_X',
                        default='data/age_regression/X_test.npy',
                        type=str,
                        help='Path to the test data')
    parser.add_argument('-test_Y',
                        '--test_Y',
                        default=None,
                        type=str,
                        help='Path to the true data')
    parser.add_argument('-hs',
                        '--hidden_size',
                        default=64,
                        type=int,
                        help='Hidden_size which using in pca dimensionality reduction operation')
    parser.add_argument('-kec',
                        '--kernal_size_conv',
                        default=7,
                        type=int,
                        help='Kernal_size which applied to convolutional layers')
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

    X_test = np.load(args.test_X)
    Y_test = np.load(args.test_Y) if args.test_Y is not None else args.test_Y
    phy_embedding = np.load("data/age_regression/phy_embedding_age.npy")

    # hac_index = hac(D)
    # phy_embedding = phy_embedding[hac_index,:]
    # X_train = X_train[:,hac_index]
    # X_eval = X_eval[:,hac_index]

    #plot_age(train_losses, val_losses, val_r2s, title='The DeepPhy model prediction on age dataset: train and test Loss/R2')


    # 绘制长度分布图
    # len_train = [len(np.nonzero(X_train[idx])[0]) for idx in range(len(X_train))]
    # plt.bar([i for i in range(len(len_train))], sorted(len_train))
    if Y_test is not None and Y_test.any():
        test_preds,test_r2 = test(X_test, Y_test, phy_embedding)
        print(f"test_r2:",test_r2)
    else:
        Y_test = np.random.uniform(-2, 2, size=X_test.shape[0])
        test_preds,_ = test(X_test, Y_test, phy_embedding)
        print(f"test_preds:",test_preds)