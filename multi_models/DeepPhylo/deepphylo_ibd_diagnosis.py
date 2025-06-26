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
                y_pred_val =model(batch['X'], batch['nonzero_indices'])
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

def shuffle_feature_embeddings(embedding_matrix, proportion=0.0):
    """
    Shuffle a given proportion of feature embeddings in the embedding matrix.
    
    Args:
    embedding_matrix (np.ndarray): The original feature embedding matrix of shape (num_features, embedding_dim).
    proportion (float): The proportion of features to shuffle, must be between 0 and 1.
    
    Returns:
    np.ndarray: The embedding matrix with a proportion of features shuffled.
    """
    # Validate input
    assert 0 <= proportion <= 1, "Proportion must be between 0 and 1"
    
    if proportion == 0:
        return embedding_matrix
    
    num_features, _ = embedding_matrix.shape
    
    # Step 1: Randomly select a specific proportion of feature indices
    num_features_to_shuffle = int(num_features * proportion)
    print(f'Number of features to shuffle: {num_features_to_shuffle}')
    candidate_indices = np.random.choice(num_features, num_features_to_shuffle, replace=False)
    
    # Step 2: Shuffle the candidate indices
    shuffled_indices = np.random.permutation(candidate_indices)
    
    # Step 3: Create a new index array where the selected indices are shuffled
    full_indices = np.arange(num_features)
    full_indices[candidate_indices] = shuffled_indices
    
    # Step 4: Reorder the embedding matrix based on the new index array
    shuffled_embedding_matrix = embedding_matrix[full_indices]
    
    return shuffled_embedding_matrix

# def reset_random_feature_embeddings(feature_embeddings, proportion=0.2, set_zero=True):
#     if proportion == 0:
#         return feature_embeddings
#     num_features, embedding_dim = feature_embeddings.shape
    
#     # 计算要重置的特征数量
#     num_reset_features = int(num_features * proportion)
    
#     # 随机选择要重置的特征索引
#     reset_indices = np.random.choice(num_features, num_reset_features, replace=False)
    
#     # 创建一个副本以进行扰乱
#     modified_feature_embeddings = feature_embeddings.copy()
    
#     if not set_zero:
#         # 生成随机embedding并重置选定的特征
#         random_embeddings = np.random.rand(num_reset_features, embedding_dim)
#         modified_feature_embeddings[reset_indices] = random_embeddings
#     else:
#         # 将选定的特征重置为零
#         modified_feature_embeddings[reset_indices] = 0
    
#     return modified_feature_embeddings

def reset_random_feature_embeddings(feature_embeddings, proportion=0.2, set_zero=True, seed=None):
    if proportion == 0:
        return feature_embeddings
    
    num_features, embedding_dim = feature_embeddings.shape
    
    # 计算要重置的特征数量
    num_reset_features = int(num_features * proportion)
    
    # 创建本地随机状态
    random_state = np.random.RandomState(seed)
    
    # 随机选择要重置的特征索引
    reset_indices = random_state.choice(num_features, num_reset_features, replace=False)
    
    # 创建一个副本以进行扰乱
    modified_feature_embeddings = feature_embeddings.copy()
    
    if not set_zero:
        # 生成随机embedding并重置选定的特征
        random_embeddings = random_state.rand(num_reset_features, embedding_dim)
        modified_feature_embeddings[reset_indices] = random_embeddings
    else:
        # 将选定的特征重置为零
        modified_feature_embeddings[reset_indices] = 0
    
    return modified_feature_embeddings


if __name__ == '__main__':
    set_seed(1234)

    parser = argparse.ArgumentParser(
        description='Command line tool for IBD diagnosis')

    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
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
                    help='proportion of feature embeddings to be reseted')
    parser.add_argument('-s',
                    '--seed',
                    default=None,
                    type=int,
                    help='random seed')
    args = parser.parse_args()
    

    # cuda visible devices
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    base_path = 'data/ibd_diagnosis/'

    # 创建一个空的字典来存储加载的数组
    X_arrays = {}
    Y_arrays = {}
    # 使用循环来加载数组
    for i in range(1):
        X_arrays['X_' + str(i)] = np.load(os.path.join(base_path, 'X_' + str(i) + '.npy'))
        Y_arrays['Y_' + str(i)] = np.load(os.path.join(base_path, 'Y_' + str(i) + '.npy'))

    phy_embedding = np.load('data/ibd_diagnosis/embedding.npy')
    
    # Generate all combinations of hyperparameters
    hidden_size = args.hidden_size
    kernal_size_conv = args.kernal_size_conv
    kernel_size_pool = args.kernal_size_pool
    dropout_conv = args.dropout
    portion = args.portion_shuffle
    seed = args.seed
    phy_embedding = reset_random_feature_embeddings(phy_embedding, proportion=portion, set_zero=True, seed=seed)
    # 定义激活函数
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
        Y_train = np.concatenate([Y_arrays['Y_' + str(j)] for j in range(15) if j != i])
        X_eval = X_arrays['X_' + str(i)]
        Y_eval = Y_arrays['Y_' + str(i)]
        print(f'Number of validation samples: {len(Y_eval)}, +: {sum(Y_eval)}, -: {len(Y_eval) - sum(Y_eval)}')
        train_losses, val_losses, val_true_labels, val_pred_labels = train(X_train, Y_train, X_eval, Y_eval, phy_embedding, batch_size=batch_size, lr=lr, hidden_size=hidden_size, kernal_size_conv=kernal_size_conv, kernel_size_pool=kernel_size_pool, dropout_conv=dropout_conv, activation=activation)

        best_epoch = select_best_epoch(val_losses)
        # 将最佳epoch的预测保存
        all_true.append(val_true_labels[best_epoch])
        all_preds.append(val_pred_labels[best_epoch])

    # 计算性能指标
    all_true = np.concatenate(all_true)
    all_preds = np.concatenate(all_preds)
    metric_dict = compute_metrics(all_true, all_preds)
    print(metric_dict)
    # with open('ibd_diagnosis_result.txt', 'a') as f:
    #     f.write(f'hidden_size: {hidden_size}, kernal_size_conv: {kernal_size_conv}, kernel_size_pool: {kernel_size_pool}, dropout_conv: {dropout_conv}, activation: {args.activation}, lr: {lr}, batch_size: {batch_size}, portion_shuffle: {args.portion_shuffle}\n')
    #     f.write(str(metric_dict) + '\n')


    
    
