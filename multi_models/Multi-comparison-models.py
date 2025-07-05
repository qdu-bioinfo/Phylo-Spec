import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from Bio import Phylo
from DeepPhylo.deepphylo.model import DeepPhylo_ibd as DeepPhylo
from DeepPhylo.deepphylo.pre_dataset import DeepPhyDataset
from data_preprocessing.data_processing import load_and_preprocess_data, match_leaf_nodes, assign_unique_names
import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn import preprocessing
from ete3 import Tree
from torch import nn, optim
import warnings
import torch
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import argparse
warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# model3-PMCNN
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

# model3-PMCNN
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
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.fc1 = nn.Linear(704, 64)
        self.fc2 = nn.Linear(64, 2)

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


# model3-PMCNN
def int_str(clustered_groups):
    MyFea_df = pd.read_csv(clustered_groups)
    row_list = [[] for _ in range(MyFea_df.shape[0])]
    for index, row in MyFea_df.iterrows():
        row_list[index] = list(row)
        row_list[index] = list(map(str, row_list[index]))
    return row_list


# model3-PMCNN
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


# model1-MetaDR
def run_model_1_train(abundance_file,tree_file,in_feature):

    abundance_file = abundance_file
    tree_file = tree_file
    output_prefix = 'output'
    output_xlsx = "prediction_results.xlsx"
    output_label_map = "label_encoding_mapping.csv"
    n_splits = 5
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(abundance_file, index_col=0)
    label_col = df.columns[-1]
    X = df.iloc[:, :-1]
    y = df[label_col]

    tree = Tree(tree_file, format=1)
    level_order = [leaf.name for leaf in tree.traverse("levelorder") if leaf.is_leaf()]
    post_order = [leaf.name for leaf in tree.traverse("postorder") if leaf.is_leaf()]

    taxa_level = [i for i in level_order if i in X.columns]
    taxa_post = [i for i in post_order if i in X.columns]

    Xl_raw = X[taxa_level]
    Xp_raw = X[taxa_post]

    Xl_raw[label_col] = y
    Xp_raw[label_col] = y
    Xl_raw.to_csv(f"{output_prefix}_levelorder.csv")
    Xp_raw.to_csv(f"{output_prefix}_postorder.csv")

    def transform_image(X, zigzag=False):
        X = np.array(X)
        raw_dim = X.shape[1]
        img_size = int(np.ceil(raw_dim ** 0.5))
        new_dim = img_size ** 2
        pad = np.zeros((X.shape[0], new_dim - raw_dim))
        new_X = np.hstack((X, pad)).reshape(X.shape[0], img_size, img_size)

        if zigzag:
            for img in new_X:
                for row in range(img.shape[0]):
                    if row % 2 != 0:
                        img[row] = img[row][::-1]

        new_X = np.log(new_X + 1) / np.log(4)
        flat = new_X.flatten()
        quantiles = np.quantile(flat, np.linspace(0, 1, 11))
        bins = [[quantiles[i], quantiles[i + 1]] for i in range(10)]
        color_vals = [0.1 * (i + 1) for i in range(10)]

        for i, (low, high) in enumerate(bins):
            mask = (new_X >= low) & (new_X < high)
            new_X[mask] = color_vals[i]

        return new_X[:, np.newaxis, :, :]

    class MetaDR(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Conv2d(20, 50, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_feature, 500),
                nn.ReLU(),
                nn.Linear(500, num_classes)
            )

        def forward(self, x):
            return self.fc(self.features(x))

    def train_and_predict(X_train, y_train, X_test, num_classes):
        model = MetaDR(num_classes).to(device)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    le = preprocessing.LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(np.unique(y_enc))

    pd.DataFrame({'Label': le.classes_, 'Encoded': le.transform(le.classes_)}).to_csv(output_label_map, index=False)

    Xl = transform_image(Xl_raw.drop(columns=[label_col]), zigzag=False)
    Xp = transform_image(Xp_raw.drop(columns=[label_col]), zigzag=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_per_class = {i: [] for i in range(num_classes)}
    kappa_list = []
    results = []
    sample_ids = df.index.tolist()

    start = time.time()
    for fold, (train_idx, test_idx) in enumerate(skf.split(Xl, y_enc), 1):
        print(f"\n▶ Fold {fold}")
        Xl_train, Xl_test = Xl[train_idx], Xl[test_idx]
        Xp_train, Xp_test = Xp[train_idx], Xp[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        probs_l = train_and_predict(Xl_train, y_train, Xl_test, num_classes)
        probs_p = train_and_predict(Xp_train, y_train, Xp_test, num_classes)
        probs_avg = (probs_l + probs_p) / 2

        y_pred = np.argmax(probs_avg, axis=1)
        kappa = cohen_kappa_score(y_test, y_pred)
        kappa_list.append(kappa)

        for i in range(num_classes):
            auc = roc_auc_score((y_test == i).astype(int), probs_avg[:, i])
            auc_per_class[i].append(auc)

        for i, idx in enumerate(test_idx):
            results.append({
                "SampleID": sample_ids[idx],
                "True Label": y_enc[idx],
                "Predicted Probability": probs_avg[i][1] if num_classes == 2 else probs_avg[i].tolist()
            })

    print("\n======== AUC (Per Class) ========")
    for i in range(num_classes):
        print(f"Class {i} ({le.classes_[i]}) AUC: {np.mean(auc_per_class[i]):.4f}")

    print("==================================")
    print(f"Mean Kappa Score: {np.mean(kappa_list):.4f}")
    print("==================================")
    print("Time cost: %.2f s" % (time.time() - start))

    pd.DataFrame(results).to_excel(output_xlsx, index=False)
    print(f"✅ Prediction results saved to {output_xlsx}")
    print(f"✅ Label encoding saved to {output_label_map}")


# model2-RF
def run_model_2_train(X, y, train_idx, val_idx):
    X_train, X_test = X[train_idx], X[val_idx]
    y_train, y_test = y[train_idx], y[val_idx]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_classifier = RandomForestClassifier(n_estimators=500,max_depth=4,max_features="log2",random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)

    return auc_score,y_test,y_pred_prob


# model3-PMCNN
def run_model_3_train(X, y, train_idx, val_idx):
    X_train, X_test = X[train_idx], X[val_idx]
    y_train, y_test = y[train_idx], y[val_idx]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_smote = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = MyDataset(X_train_smote[:, :X_train_smote.shape[1] // 4],
                              X_train_smote[:, X_train_smote.shape[1] // 4:X_train_smote.shape[1] // 2],
                              X_train_smote[:, X_train_smote.shape[1] // 2:3 * X_train_smote.shape[1] // 4],
                              X_train_smote[:, 3 * X_train_smote.shape[1] // 4:], y_train)
    test_dataset = MyDataset(X_test[:, :X_test.shape[1] // 4],
                             X_test[:, X_test.shape[1] // 4:X_test.shape[1] // 2],
                             X_test[:, X_test.shape[1] // 2:3 * X_test.shape[1] // 4],
                             X_test[:, 3 * X_test.shape[1] // 4:], y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(10):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            x_train1, x_train2, x_train3, x_train4 = inputs
            y_pred = model(x_train1, x_train2, x_train3, x_train4)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            x_test1, x_test2, x_test3, x_test4 = inputs
            outputs = model(x_test1, x_test2, x_test3, x_test4)

            probs = torch.softmax(outputs, dim=1)
            preds = probs[:, 1]

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    auc_score = roc_auc_score(all_labels, all_preds)

    return auc_score, all_labels, all_preds


# model4-DeepPhylo
def run_model_4_train(X, y, train_idx, val_idx, phy_embedding):

    X_train, X_eval = X[train_idx], X[val_idx]
    y_train, y_eval = y[train_idx], y[val_idx]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)


    train_losses, val_losses, val_true_labels, val_pred_labels = train(
        X_train, y_train, X_eval, y_eval, phy_embedding,
        train_batch_size=128, val_batch_size=64,
        lr=0.0001, hidden_size=32,
        kernal_size_conv=13, kernel_size_pool=4,
        dropout_conv=0.2, activation=nn.LeakyReLU())
    best_epoch = select_best_epoch(val_losses)

    return val_true_labels[best_epoch], val_pred_labels[best_epoch]

# model4-DeepPhylo
def train(X_train, Y_train, X_eval, Y_eval, phy_embedding,
          train_batch_size=128, val_batch_size=64,
          lr=1e-4, hidden_size=32, kernal_size_conv=13,
          kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.BCELoss()

    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.custom_collate_fn
    )

    val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=train_dataset.custom_collate_fn
    )

    model = DeepPhylo(
        hidden_size=hidden_size,
        embeddings=train_dataset.embeddings,
        kernel_size_conv=kernal_size_conv,
        kernel_size_pool=kernel_size_pool,
        dropout_conv=dropout_conv,
        activation=activation
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    epochs = 10
    patience = 1
    best_val_loss = float("inf")
    counter = 0
    train_losses = []
    val_losses = []
    val_pred_labels = []
    val_true_labels = []

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            y_pred_train = model(batch['X'], batch['nonzero_indices']).squeeze(dim=1)
            loss_train = criterion(y_pred_train, batch['y'])
            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item() * batch['X'].size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        y_val = []
        with torch.no_grad():
            for batch in val_loader:
                y_val.append(batch['y'].numpy())
                batch = {key: val.to(device) for key, val in batch.items()}
                y_pred_val = model(batch['X'], batch['nonzero_indices']).squeeze(dim=1)
                loss_val = criterion(y_pred_val, batch['y'])
                val_loss += loss_val.item() * batch['X'].size(0)
                val_preds.append(y_pred_val.detach().cpu().numpy())
        val_loss /= len(val_loader.dataset)

        y_val = np.concatenate(y_val)
        val_preds = np.concatenate(val_preds)
        val_pred_labels.append(val_preds)
        val_true_labels.append(y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

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


# model5-CNN
import torch.nn as nn

def convolution_block(in_channels, out_channels, kernel_size=3, padding="same"):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.Dropout(p=0.5),
        nn.ReLU(inplace=False),
    )

# model5-CNN
class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()

        self.conv1 = convolution_block(1, 16, kernel_size=3)
        self.conv2 = convolution_block(16, 8, kernel_size=5)
        self.conv3 = convolution_block(8, 4, kernel_size=7)
        self.conv4 = convolution_block(4, 2, kernel_size=9)

        self.fc1 = nn.Linear(2*input_dim, 128)
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

# model5-CNN
def run_cnn_train(X, y, train_idx, val_idx, input_dim, output_dim, epochs):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    model = CNNModel(input_dim=input_dim, output_dim=output_dim).to('cpu')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    val_data = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(1), y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    model.eval()
    with torch.no_grad():
        all_pred_probs = []
        all_true_labels = []
        for X_batch, y_batch in val_loader:
            y_pred_val = model(X_batch)

            y_pred_prob = torch.sigmoid(y_pred_val).cpu().numpy()
            all_pred_probs.extend(y_pred_prob)
            all_true_labels.extend(y_batch.cpu().numpy())

    y_pred_prob = np.array(all_pred_probs)
    y_true = np.array(all_true_labels)
    auc_score = roc_auc_score(y_true, y_pred_prob)

    return auc_score, y_true, y_pred_prob

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all 5 microbiome models with evaluation.')

    parser.add_argument('-c', '--csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('-t', '--tree', type=str, required=True, help='Path to Newick tree file')
    parser.add_argument('-l', '--list', type=str, required=True, help='Path to PMCNN feature list CSV')
    parser.add_argument('-npy', nargs=3, type=str, required=True, metavar=('X', 'Y', 'EMBEDDING'),
                        help='Paths to DeepPhylo input .npy files: X.npy Y.npy embedding.npy')

    args = parser.parse_args()

    set_seed(42)

    # Arguments
    csv_path = args.csv
    newick_path = args.tree
    list_path = args.list
    X_path, y_path, embedding_path = args.npy

    # Model 1 MetaDR
    tree = Phylo.read(newick_path, 'newick')
    tree = assign_unique_names(tree)
    X1, y1, encoder1, data1 = load_and_preprocess_data(csv_path, tree)
    run_model_1_train(csv_path, newick_path, in_feature=1250)

    # Model 2 RF
    df = pd.read_csv(csv_path)
    X2 = df.iloc[:, 1:-1].values
    y2 = df.iloc[:, -1].values
    encoder2 = LabelEncoder()
    y2_encoded = encoder2.fit_transform(y2)

    # Model 3 PM-CNN
    data3 = pd.read_csv(csv_path)
    My_list = int_str(list_path)
    data_list, y3 = load_data(data3.iloc[:, 1:-1], data3.iloc[:, -1], My_list)
    X3 = torch.cat(data_list, dim=1)

    # Model 4 DeepPhylo
    X4 = np.load(X_path, allow_pickle=True)
    y4 = np.load(y_path, allow_pickle=True)
    phy_embedding = np.load(embedding_path)

    # Model 5 CNN
    df = pd.read_csv(csv_path)
    X5 = df.iloc[:, 1:-1].values
    num_columns = df.iloc[:, 1:-1].shape[1]
    y5 = df.iloc[:, -1].values
    encoder5 = LabelEncoder()
    y5_encoded = encoder5.fit_transform(y5)

    # Store AUCs
    auc_scores_model_2 = []
    auc_scores_model_3 = []
    auc_scores_model_4 = []
    auc_scores_model_5 = []

    all_fold_roc_data = {
        'model_2': {'true_labels': [], 'pred_scores': []},
        'model_3': {'true_labels': [], 'pred_scores': []},
        'model_4': {'true_labels': [], 'pred_scores': []},
        'model_5': {'true_labels': [], 'pred_scores': []}
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X1, y1)):
        print(f"Running Fold {fold + 1}...")

        # Model 2
        auc2, y_test, y_pred2 = run_model_2_train(X2, y2_encoded, train_idx, val_idx)
        all_fold_roc_data['model_2']['true_labels'].append(y_test)
        all_fold_roc_data['model_2']['pred_scores'].append(y_pred2)
        auc_scores_model_2.append(auc2)

        # Model 3
        auc3, y3_true, y3_pred = run_model_3_train(X3, y3, train_idx, val_idx)
        all_fold_roc_data['model_3']['true_labels'].append(y3_true)
        all_fold_roc_data['model_3']['pred_scores'].append(y3_pred)
        auc_scores_model_3.append(auc3)

        # Model 4
        train_idx_4 = [i for i in train_idx if i < len(X4)]
        val_idx_4 = [i for i in val_idx if i < len(X4)]
        y4_true, y4_pred = run_model_4_train(X4, y4, train_idx_4, val_idx_4, phy_embedding)
        all_fold_roc_data['model_4']['true_labels'].append(y4_true)
        all_fold_roc_data['model_4']['pred_scores'].append(y4_pred)
        auc4 = roc_auc_score(y4_true, y4_pred)
        auc_scores_model_4.append(auc4)

        # Model 5
        auc5, y5_true, y5_pred = run_cnn_train(X5, y5_encoded, train_idx, val_idx, input_dim=num_columns, output_dim=1, epochs=10)
        all_fold_roc_data['model_5']['true_labels'].append(y5_true)
        all_fold_roc_data['model_5']['pred_scores'].append(y5_pred)
        auc_scores_model_5.append(auc5)

    print(f"Model 2 Mean AUC: {np.mean(auc_scores_model_2):.4f}")
    print(f"Model 3 Mean AUC: {np.mean(auc_scores_model_3):.4f}")
    print(f"Model 4 Mean AUC: {np.mean(auc_scores_model_4):.4f}")
    print(f"Model 5 Mean AUC: {np.mean(auc_scores_model_5):.4f}")
