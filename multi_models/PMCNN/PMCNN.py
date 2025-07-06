import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from torch import optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import random

# -------- Utility functions --------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def int_str(clustered_groups):
    MyFea_df = pd.read_csv(clustered_groups)
    row_list = [[] for _ in range(MyFea_df.shape[0])]
    for index, row in MyFea_df.iterrows():
        row_list[index] = list(map(str, list(row)))
    return row_list

def load_data(otu_table_path, meta_file_path, My_list):
    X = pd.read_csv(otu_table_path, index_col=0)
    y = pd.read_csv(meta_file_path, index_col=0)
    y = y.iloc[:, 0].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y.ravel())
    y = torch.LongTensor(y)

    x1 = X[My_list[0]]
    x2 = X[My_list[1]]
    x3 = X[My_list[2]]
    x4 = X[My_list[3]]

    data_list = [x1, x2, x3, x4]
    for i in range(len(data_list)):
        data_list[i] = torch.FloatTensor(np.array(data_list[i], dtype=np.float32))

    return data_list, y

# -------- Model & Dataset --------
class MyDataset(Dataset):
    def __init__(self, x1, x2, x3, x4, y):
        self.x1, self.x2, self.x3, self.x4, self.y = x1, x2, x3, x4, y
    def __len__(self): return len(self.y)
    def __getitem__(self, index): return (self.x1[index], self.x2[index], self.x3[index], self.x4[index]), self.y[index]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv2_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv3_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv4_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.conv4_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=7, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(2816, 64)
        self.fc2 = nn.Linear(64, 2)

    def conv_block(self, x, conv1, conv2):
        x = F.tanh(conv1(x))
        x = nn.BatchNorm1d(16)(x)
        x = F.tanh(conv2(x))
        x = nn.BatchNorm1d(16)(x)
        return x

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv_block(x1.unsqueeze(1), self.conv1_1, self.conv1_2).view(x1.size(0), -1)
        x2 = self.conv_block(x2.unsqueeze(1), self.conv2_1, self.conv2_2).view(x2.size(0), -1)
        x3 = self.conv_block(x3.unsqueeze(1), self.conv3_1, self.conv3_2).view(x3.size(0), -1)
        x4 = self.conv_block(x4.unsqueeze(1), self.conv4_1, self.conv4_2).view(x4.size(0), -1)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.tanh(self.bn1(self.fc1(x)))
        return F.softmax(self.fc2(x), dim=1)

# -------- Training & Evaluation --------
def train_model(model, criterion, optimizer, train_loader, epoch):
    model.train()
    for _ in range(epoch):
        for inputs, labels in train_loader:
            x1, x2, x3, x4 = inputs
            y_pred = model(x1, x2, x3, x4)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            x1, x2, x3, x4 = inputs
            outputs = model(x1, x2, x3, x4)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    auc_score = roc_auc_score(all_labels, all_probs[:, 1]) if len(np.unique(all_labels)) > 1 else float('nan')

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='weighted'),
        "recall": recall_score(all_labels, all_preds, average='weighted'),
        "f1": f1_score(all_labels, all_preds, average='weighted'),
        "auc": auc_score,
        "kappa": cohen_kappa_score(all_labels, all_preds)
    }

# -------- Main Entry --------
def main(csv_path, list_path):
    set_seed(42)
    My_list = int_str(list_path)

    data = pd.read_csv(csv_path)
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, [-1]]
    X.to_csv("temp_features.csv")
    y.to_csv("temp_labels.csv")

    data_list, y_tensor = load_data("temp_features.csv", "temp_labels.csv", My_list)
    X_cat = torch.cat(data_list, dim=1)

    X_train, X_test, y_train, y_test = train_test_split(X_cat, y_tensor, test_size=0.3, stratify=y_tensor, random_state=35)

    smote = SMOTE(random_state=42)
    X_train_np, y_train_np = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)

    train_dataset = MyDataset(X_train_tensor[:, :X_train_tensor.shape[1]//4],
                              X_train_tensor[:, X_train_tensor.shape[1]//4:X_train_tensor.shape[1]//2],
                              X_train_tensor[:, X_train_tensor.shape[1]//2:3*X_train_tensor.shape[1]//4],
                              X_train_tensor[:, 3*X_train_tensor.shape[1]//4:], y_train_tensor)

    test_dataset = MyDataset(X_test_tensor[:, :X_test_tensor.shape[1]//4],
                             X_test_tensor[:, X_test_tensor.shape[1]//4:X_test_tensor.shape[1]//2],
                             X_test_tensor[:, X_test_tensor.shape[1]//2:3*X_test_tensor.shape[1]//4],
                             X_test_tensor[:, 3*X_test_tensor.shape[1]//4:], y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    train_model(model, criterion, optimizer, train_loader, epoch=10)
    results = evaluate_model(model, test_loader)

    print(f"Test AUC: {results['auc']:.4f}, Test Kappa: {results['kappa']:.4f}")

# -------- CLI --------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PM-CNN on microbial data.")
    parser.add_argument("-c", "--csv", required=True, help="Path to the input CSV file (feature+label).")
    parser.add_argument("-list", required=True, help="Path to the CSV file containing feature groupings.")

    args = parser.parse_args()
    main(args.csv, args.list)
