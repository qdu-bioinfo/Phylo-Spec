
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
import warnings
import os
import argparse

warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convolution_block(in_channels, out_channels, kernel_size=3, padding="same"):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.Dropout(p=0.5),
        nn.ReLU(inplace=False),
    )


class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()

        self.conv1 = convolution_block(1, 16, kernel_size=3)
        self.conv2 = convolution_block(16, 8, kernel_size=5)
        self.conv3 = convolution_block(8, 4, kernel_size=7)
        self.conv4 = convolution_block(4, 2, kernel_size=9)

        self.fc1 = nn.Linear(2 * input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def run_cnn_train(X, y, train_idx, val_idx, input_dim, output_dim, batch_size, epochs):

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]


    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)


    model = CNNModel(input_dim=input_dim, output_dim=output_dim).to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)


    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


    model.eval()
    with torch.no_grad():
        y_pred_val = model(X_val)

    pred_probs = torch.softmax(y_pred_val, dim=1).cpu().numpy()

    auc_score = roc_auc_score(y_val.cpu().numpy(), pred_probs, multi_class="ovr", average="macro")
    return auc_score, y_val.cpu().numpy(), pred_probs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN model on input CSV.')
    parser.add_argument('-c', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('-o', type=str, default='model5_results', help='Directory to save results')

    args = parser.parse_args()

    set_seed(42)

    output_dir = args.o
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(args.c)
    sample_names = df.iloc[:, 0].values
    X5 = df.iloc[:, 1:-1].values
    num_columns = df.iloc[:, 1:-1].shape[1]
    y5 = df.iloc[:, -1].values

    encoder5 = LabelEncoder()
    y5_encoded = encoder5.fit_transform(y5)
    num_classes = len(np.unique(y5_encoded))
    class_names = encoder5.classes_

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores_model_5 = []
    fold_kappas = []

    all_sample_names = []
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []

    fold_idx = 1
    for train_idx, val_idx in skf.split(X5, y5_encoded):
        print(f"Running Fold {fold_idx} for Model 5...")

        fold_sample_names = sample_names[val_idx]

        auc_score, true_labels, pred_probs = run_cnn_train(
            X5, y5_encoded, train_idx, val_idx,
            input_dim=num_columns,
            output_dim=num_classes,
            batch_size=1024,
            epochs=5
        )
        auc_scores_model_5.append(auc_score)

        pred_classes = np.argmax(pred_probs, axis=1)

        kappa = cohen_kappa_score(true_labels, pred_classes)
        fold_kappas.append(kappa)
        print(f"Fold {fold_idx} AUC: {auc_score:.4f}, Kappa: {kappa:.4f}")

        all_sample_names.extend(fold_sample_names)
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_classes)
        all_pred_probs.extend(pred_probs)

        fold_idx += 1

    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    all_pred_probs = np.array(all_pred_probs)

    overall_auc = roc_auc_score(all_true_labels, all_pred_probs, multi_class="ovr", average="macro")
    overall_kappa = cohen_kappa_score(all_true_labels, all_pred_labels)
    avg_kappa = np.mean(fold_kappas)

    print(f"\nMean AUC for Model 5: {overall_auc:.4f}")
    print(f"Average Cohen's Kappa across folds: {avg_kappa:.4f}")
    print(f"Overall Cohen's Kappa: {overall_kappa:.4f}")
