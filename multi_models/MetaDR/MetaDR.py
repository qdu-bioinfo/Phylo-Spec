import argparse
import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from ete3 import Tree
import torch
from torch import nn, optim
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

parser = argparse.ArgumentParser(description="Run phylogenetic model with input files.")
parser.add_argument('-c', '--abundance', required=True, help='Path to abundance CSV file')
parser.add_argument('-t', '--tree', required=True, help='Path to Newick tree file (.nwk)')
args = parser.parse_args()


abundance_file = args.c
tree_file = args.t
output_prefix = 'output'
output_xlsx = "prediction_results.xlsx"
output_label_map = "label_encoding_mapping.csv"
n_splits = 5
epochs = 1
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
    bins = [[quantiles[i], quantiles[i+1]] for i in range(10)]
    color_vals = [0.1 * (i+1) for i in range(10)]

    for i, (low, high) in enumerate(bins):
        mask = (new_X >= low) & (new_X < high)
        new_X[mask] = color_vals[i]

    return new_X[:, np.newaxis, :, :]

class SimpleCNN(nn.Module):
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
            nn.Linear(1800, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )

    def forward(self, x):
        return self.fc(self.features(x))

def train_and_predict(X_train, y_train, X_test, num_classes):
    model = SimpleCNN(num_classes).to(device)
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


print("\n======== AUC (Per Class) ========")
for i in range(num_classes):
    print(f"Class {i} ({le.classes_[i]}) AUC: {np.mean(auc_per_class[i]):.4f}")

print("==================================")
print(f"Mean Kappa Score: {np.mean(kappa_list):.4f}")
print("==================================")
print("Time cost: %.2f s" % (time.time() - start))

