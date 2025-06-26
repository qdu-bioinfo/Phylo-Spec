# PM-CNN
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from torch import optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

torch.autograd.set_detect_anomaly(True)

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
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=6, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=6, padding=1)
        self.conv2_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=6, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=6, padding=1)
        self.conv3_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=6, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=6, padding=1)
        self.conv4_1 = nn.Conv1d(1, out_channels=16, kernel_size=8, stride=6, padding=1)
        self.conv4_2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=6, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.fc1 = nn.Linear(2816, 64)  # Adjust based on input size after conv layers
        self.fc2 = nn.Linear(64, 2)  # Number of classes, adjust based on dataset

    def conv_block(self, x, conv1, conv2):
        x = F.tanh(conv1(x))
        x = nn.BatchNorm1d(num_features=16)(x)
        x = F.tanh(conv2(x))
        x = nn.BatchNorm1d(num_features=16)(x)
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
        x = self.bn1(x)
        x = F.tanh(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x

def int_str(clustered_groups):
    MyFea_df = pd.read_csv(clustered_groups)
    row_list = [[] for _ in range(MyFea_df.shape[0])]
    for index, row in MyFea_df.iterrows():
        row_list[index] = list(row)
        row_list[index] = list(map(str, row_list[index]))
    return row_list

def load_data(otu_table, meta_file, My_list):
    X = pd.read_csv(otu_table, index_col=0)
    y = pd.read_csv(meta_file, index_col=0)
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
        data_list[i] = np.array(data_list[i], dtype=np.float32)
        data_list[i] = torch.FloatTensor(data_list[i])

    return data_list, y

def train_model(model, criterion, optimizer, train_loader, epoch):
    model.train()
    for epoch in range(epoch):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            x_train1, x_train2, x_train3, x_train4 = inputs
            y_pred = model(x_train1, x_train2, x_train3, x_train4)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            x_test1, x_test2, x_test3, x_test4 = inputs
            outputs = model(x_test1, x_test2, x_test3, x_test4)
            probs = F.softmax(outputs, dim=1)
            preds = torch.max(outputs, dim=1)[1]
            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    unique_classes = np.unique(all_labels)
    if len(unique_classes) == 1:
        print("Only one class present in y_true. ROC AUC score is not defined in that case.")
        auc_score = float('nan')
    else:
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)

    print("Test AUC: {:.4f}".format(auc_score))
    print("Test Kappa: {:.4f}".format(kappa))


    return accuracy, precision, recall, f1, auc_score, kappa

def main():
    data = pd.read_csv("csv")
    My_list = int_str('csv')


    data_list, y = load_data(data.iloc[:,1:-1], data.iloc[:,-1], My_list)
    X = torch.cat(data_list, dim=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=35)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_smote = scaler.fit_transform(X_train_smote)
    X_test = scaler.transform(X_test)

    X_train_smote = torch.tensor(X_train_smote, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = MyDataset(X_train_smote[:, :X_train_smote.shape[1] // 4],
                              X_train_smote[:, X_train_smote.shape[1] // 4:X_train_smote.shape[1] // 2],
                              X_train_smote[:, X_train_smote.shape[1] // 2:3 * X_train_smote.shape[1] // 4],
                              X_train_smote[:, 3 * X_train_smote.shape[1] // 4:], y_train_smote)
    test_dataset = MyDataset(X_test[:, :X_test.shape[1] // 4],
                             X_test[:, X_test.shape[1] // 4:X_test.shape[1] // 2],
                             X_test[:, X_test.shape[1] // 2:3 * X_test.shape[1] // 4],
                             X_test[:, 3 * X_test.shape[1] // 4:], y_test)


    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    train_model(model, criterion, optimizer, train_loader, 5)

    accuracy, precision, recall, f1, auc_score, kappa = evaluate_model(model, test_loader)

    print(f"Test AUC: {auc_score:.4f}, Test Kappa: {kappa:.4f}")

if __name__ == '__main__':
    main()
