import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader

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


def test(X_test, Y_test, phy_embedding, batch_size=32, lr=1e-4, hidden_size=32, kernal_size_conv=13, kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU(),i=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCELoss()
    batch_size = batch_size
    # Create DataLoader for training and validation data
    test_dataset = DeepPhyDataset(phy_embedding, X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=test_dataset.custom_collate_fn)
    model = DeepPhylo(hidden_size=hidden_size, 
                    embeddings=test_dataset.embeddings, 
                    kernel_size_conv=kernal_size_conv, 
                    kernel_size_pool=kernel_size_pool,
                    dropout_conv=dropout_conv,
                    activation=activation).to(device)

    # Testing
    model.load_state_dict(torch.load(f"data/ibd_diagnosis/best_model_{i}.pth"))
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for batch in test_loader:
            test_true.append(batch['y'].numpy())
            batch = {key: val.to(device) for key, val in batch.items()}
            y_pred_test =model(batch['X'], batch['nonzero_indices'])
            y_pred_test = y_pred_test.squeeze(dim=1)
            test_preds.append(y_pred_test.detach().cpu().numpy())
    # Calculate validation R2
    test_true = np.concatenate(test_true)
    test_preds = np.concatenate(test_preds)

    # Early stopping

    del model
    return test_true, test_preds


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
                    help='dropout rate of convolutional layers')
    args = parser.parse_args()
    

    # cuda visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    base_path = 'data/ibd_diagnosis/'

    # 创建一个空的字典来存储加载的数组
    X_arrays = {}
    Y_arrays = {}
    # 使用循环来加载数组
    for i in range(15):
        X_arrays['X_' + str(i)] = np.load(os.path.join(base_path, 'X_' + str(i) + '.npy'))
        Y_arrays['Y_' + str(i)] = np.load(os.path.join(base_path, 'Y_' + str(i) + '.npy'))

    phy_embedding = np.load('data/ibd_diagnosis/embedding.npy')
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

    # 使用循环来实现leave one dataset out的评估
    all_preds = []
    all_true = []
    for i in range(15):
        # 创建训练集和验证集
        X_train = np.concatenate([X_arrays['X_' + str(j)] for j in range(15) if j != i])
        X_test = X_arrays['X_' + str(i)]
        Y_test = Y_arrays['Y_' + str(i)]
        phy_embedding, _ , X_test = random_shuffle(phy_embedding,  X_train,X_test,portion=portion)
        print(f'Number of validation samples: {len(Y_test)}, +: {sum(Y_test)}, -: {len(Y_test) - sum(Y_test)}')
        test_true, test_preds = test(X_test, Y_test, phy_embedding, batch_size=batch_size, lr=lr, hidden_size=hidden_size, kernal_size_conv=kernal_size_conv, kernel_size_pool=kernel_size_pool, dropout_conv=dropout_conv, activation=activation,i=i)

        # 将最佳epoch的预测保存
        all_true.append(test_true)
        all_preds.append(test_preds)

    # 计算性能指标
    all_true = np.concatenate(all_true)
    all_preds = np.concatenate(all_preds)
    metric_dict = compute_metrics(all_true, all_preds)
    print(metric_dict)
    # with open('ibd_diagnosis_result.txt', 'a') as f:
    #     f.write(f'hidden_size: {hidden_size}, kernal_size_conv: {kernal_size_conv}, kernel_size_pool: {kernel_size_pool}, dropout_conv: {dropout_conv}, activation: {args.activation}, lr: {lr}, batch_size: {batch_size}, portion_shuffle: {args.portion_shuffle}\n')
    #     f.write(str(metric_dict) + '\n')


    
    
