import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve, auc, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyDataset(Dataset):
    def __init__(self, x1, x2, x3, x4, y):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.x1[index], self.x2[index], self.x3[index], self.x4[index]), self.y[index]


class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv2_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv3_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv4_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv4_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.fc1 = nn.Linear(2368, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def conv_block(self, x, conv1, conv2):
        x = torch.relu(conv1(x))
        x = torch.relu(conv2(x))
        return x

    def forward(self, x1, x2, x3, x4):

        x1 = x1.reshape(-1, 1, x1.size(1))
        x2 = x2.reshape(-1, 1, x2.size(1))
        x3 = x3.reshape(-1, 1, x3.size(1))
        x4 = x4.reshape(-1, 1, x4.size(1))

        x1 = self.conv_block(x1, self.conv1_1, self.conv1_2)
        x2 = self.conv_block(x2, self.conv2_1, self.conv2_2)
        x3 = self.conv_block(x3, self.conv3_1, self.conv3_2)
        x4 = self.conv_block(x4, self.conv4_1, self.conv4_2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def int_str(clustered_groups):
    MyFea_df = pd.read_csv(clustered_groups)
    row_list = [[] for _ in range(MyFea_df.shape[0])]
    for index, row in MyFea_df.iterrows():
        row_list[index] = list(row)
        row_list[index] = list(map(str, row_list[index]))
    return row_list


def load_data(otu_table, meta_file, My_list):
    X = otu_table
    y = meta_file
    encoder = LabelEncoder()
    y = encoder.fit_transform(y.ravel())
    y = torch.LongTensor(y)

    x1 = X[My_list[0]]
    x2 = X[My_list[1]]
    x3 = X[My_list[2]]
    x4 = X[My_list[3]]

    data_list = [x1, x2, x3, x4]
    for i in range(len(data_list)):
        data_list[i] = np.array(data_list[i], dtype=np.float32)
        data_list[i] = torch.FloatTensor(data_list[i])

    return data_list, y


def run_model_3_train(X, y, train_idx, val_idx):

    X_train, X_test = X[train_idx], X[val_idx]
    y_train, y_test = y[train_idx], y[val_idx]

    smote = SMOTE(random_state=42)
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_train_np, y_train_np = smote.fit_resample(X_train_np, y_train_np)

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test.numpy())

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)

    num_classes = int(np.max(y_train_np)) + 1

    train_dataset = MyDataset(
        X_train_tensor[:, :X_train_tensor.shape[1] // 4],
        X_train_tensor[:, X_train_tensor.shape[1] // 4:X_train_tensor.shape[1] // 2],
        X_train_tensor[:, X_train_tensor.shape[1] // 2:3 * X_train_tensor.shape[1] // 4],
        X_train_tensor[:, 3 * X_train_tensor.shape[1] // 4:],
        y_train_tensor
    )
    test_dataset = MyDataset(
        X_test_tensor[:, :X_test_tensor.shape[1] // 4],
        X_test_tensor[:, X_test_tensor.shape[1] // 4:X_test_tensor.shape[1] // 2],
        X_test_tensor[:, X_test_tensor.shape[1] // 2:3 * X_test_tensor.shape[1] // 4],
        X_test_tensor[:, 3 * X_test_tensor.shape[1] // 4:],
        y_test
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Net(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

    model.train()
    for epoch in range(10):
        for inputs, labels in train_loader:
            x1, x2, x3, x4 = inputs
            outputs = model(x1, x2, x3, x4)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            x1, x2, x3, x4 = inputs
            outputs = model(x1, x2, x3, x4)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    auc_score = roc_auc_score(all_labels, all_preds, multi_class="ovr", average="macro")
    return auc_score, all_labels, all_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process input CSV files.")
    parser.add_argument('-c', required=True, help='Path to main CSV file')
    parser.add_argument('-list', required=True, help='Path to list CSV file')
    args = parser.parse_args()
    set_seed(42)

    csv_path = args.c
    list_path = args.list

    data3 = pd.read_csv(csv_path)
    sample_names = data3.iloc[:, 0].values
    My_list = int_str(list_path)
    data_list, y3 = load_data(data3.iloc[:, 1:-1], data3.iloc[:, -1], My_list)

    X3 = torch.cat(data_list, dim=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores_model_3 = []
    fold_kappas = []
    aggregated_true = []
    aggregated_preds = []
    aggregated_sample_names = []

    fold_idx = 1
    for train_idx, val_idx in skf.split(X3.numpy(), y3.numpy()):
        print(f"Running Fold {fold_idx} for Model 3...")
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()

        fold_sample_names = sample_names[val_idx]

        auc_score, true_labels, pred_probs = run_model_3_train(X3, y3, train_idx, val_idx)
        auc_scores_model_3.append(auc_score)

        pred_classes = np.argmax(pred_probs, axis=1)

        kappa = cohen_kappa_score(true_labels, pred_classes)
        fold_kappas.append(kappa)
        print(f"Fold {fold_idx} AUC: {auc_score:.4f}, Kappa: {kappa:.4f}")

        aggregated_true.extend(true_labels.tolist())
        aggregated_preds.extend(pred_probs.tolist())
        aggregated_sample_names.extend(fold_sample_names)

        fold_idx += 1

    aggregated_true = np.array(aggregated_true)
    aggregated_preds = np.array(aggregated_preds)
    overall_kappa = cohen_kappa_score(aggregated_true, np.argmax(aggregated_preds, axis=1))
    avg_kappa = np.mean(fold_kappas)

    print(f"\nMean AUC for Model 3: {np.mean(auc_scores_model_3):.4f}")
    print(f"Average Cohen's Kappa across folds: {avg_kappa:.4f}")
    print(f"Overall Cohen's Kappa: {overall_kappa:.4f}")

    print("\nAll analysis completed successfully!")
