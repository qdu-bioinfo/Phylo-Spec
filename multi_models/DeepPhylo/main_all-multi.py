import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, auc, cohen_kappa_score
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from DeepPhylo.deepphylo.pre_dataset import DeepPhyDataset
from DeepPhylo.deepphylo.model import DeepPhylo_ibd as DeepPhylo


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_multiclass_roc(y_true, y_score, n_classes, filename):

    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (AUC = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

class MultiClassDeepPhylo(DeepPhylo):
    def __init__(self, hidden_size, embeddings, kernel_size_conv=13,
                 kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU(),
                 n_classes=2):
        super().__init__(hidden_size, embeddings, kernel_size_conv,
                         kernel_size_pool, dropout_conv, activation)


        self.dim_adjust = nn.Linear(1, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, X, nonzero_indices):

        features = super().forward(X, nonzero_indices)
        if features.dim() == 3:
            features = features.squeeze(-1)
        features = self.dim_adjust(features)

        return self.fc(features)

def run_model_4_train(X, y, train_idx, val_idx, phy_embedding, n_classes):

    X_train, X_eval = X[train_idx], X[val_idx]
    y_train, y_eval = y[train_idx], y[val_idx]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    train_losses, val_losses, val_true_labels, val_pred_probs = train(
        X_train, y_train, X_eval, y_eval, phy_embedding, n_classes)

    best_epoch = select_best_epoch(val_losses)

    best_true_labels = val_true_labels[best_epoch]
    best_pred_probs = val_pred_probs[best_epoch]

    best_pred_labels = np.argmax(best_pred_probs, axis=1)

    return best_true_labels, best_pred_labels, best_pred_probs

def train(X_train, Y_train, X_eval, Y_eval, phy_embedding, n_classes,
          batch_size=32, lr=1e-4, hidden_size=32, kernal_size_conv=13,
          kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.custom_collate_fn)
    val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=train_dataset.custom_collate_fn)
    model = MultiClassDeepPhylo(hidden_size=hidden_size,
                                embeddings=train_dataset.embeddings,
                                kernel_size_conv=kernal_size_conv,
                                kernel_size_pool=kernel_size_pool,
                                dropout_conv=dropout_conv,
                                activation=activation,
                                n_classes=n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    epochs = 10
    patience = 20
    best_val_loss = float("inf")
    counter = 0

    train_losses = []
    val_losses = []
    val_pred_probs = []
    val_true_labels = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            y_pred_train = model(batch['X'], batch['nonzero_indices'])
            loss_train = criterion(y_pred_train, batch['y'].long())
            loss_train.backward()
            optimizer.step()
            train_loss += loss_train.item() * batch['X'].size(0)

        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        epoch_pred_probs = []
        epoch_true_labels = []
        with torch.no_grad():
            for batch in val_loader:
                epoch_true_labels.append(batch['y'].numpy())
                batch = {key: val.to(device) for key, val in batch.items()}
                y_pred_val = model(batch['X'], batch['nonzero_indices'])
                loss_val = criterion(y_pred_val, batch['y'].long())
                val_loss += loss_val.item() * batch['X'].size(0)
                probs = torch.softmax(y_pred_val, dim=1).detach().cpu().numpy()
                epoch_pred_probs.append(probs)

        val_loss /= len(val_loader.dataset)
        epoch_true_labels = np.concatenate(epoch_true_labels)
        epoch_pred_probs = np.concatenate(epoch_pred_probs, axis=0)

        val_pred_probs.append(epoch_pred_probs)
        val_true_labels.append(epoch_true_labels)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    del model, optimizer
    return train_losses, val_losses, val_true_labels, val_pred_probs

def select_best_epoch(val_losses):
    best_epoch = np.argmin(val_losses)
    return best_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CNN training with .npy files")
    parser.add_argument('-xnpy', required=True, help='Path to X .npy file')
    parser.add_argument('-ynpy', required=True, help='Path to y .npy file')
    parser.add_argument('-embed', required=True, help='Path to embedding .npy file')
    args = parser.parse_args()

    set_seed(42)

    # Load inputs
    X4 = np.load(args.xnpy)
    y4 = np.load(args.ynpy)
    phy_embedding = np.load(args.embed)

    # Load sample names (assumes a CSV with same name as y.npy)
    y_csv_path = args.y.replace('.npy', '.csv')
    sample_names = pd.read_csv(y_csv_path).iloc[:, 0].values

    n_classes = len(np.unique(y4))
    print(f"Number of classes: {n_classes}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores_model_4 = []
    kappa_scores = []

    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    all_sample_names = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X4, y4)):
        print(f"\nRunning Fold {fold + 1} for Model 4...")

        train_idx_4 = [i for i in train_idx if i < len(X4)]
        val_idx_4 = [i for i in val_idx if i < len(X4)]

        fold_sample_names = sample_names[val_idx_4]

        true_labels, pred_labels, pred_probs = run_model_4_train(
            X4, y4, train_idx_4, val_idx_4, phy_embedding, n_classes)

        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
        all_pred_probs.append(pred_probs)
        all_sample_names.extend(fold_sample_names)

        if n_classes > 2:
            y_true_bin = label_binarize(true_labels, classes=np.arange(n_classes))
            fold_auc = roc_auc_score(y_true_bin, pred_probs, multi_class='ovr')
            plot_filename = f"roc_curve_fold_{fold + 1}.pdf"
            plot_multiclass_roc(true_labels, pred_probs, n_classes, plot_filename)
            print(f"ROC curve saved as {plot_filename}")
        else:
            fold_auc = roc_auc_score(true_labels, pred_probs[:, 1])

        auc_scores_model_4.append(fold_auc)

        fold_kappa = cohen_kappa_score(true_labels, pred_labels)
        kappa_scores.append(fold_kappa)

        print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
        print(f"Fold {fold + 1} Kappa: {fold_kappa:.4f}")

    all_pred_probs = np.concatenate(all_pred_probs, axis=0)

    results_df = pd.DataFrame({
        'Sample_Name': all_sample_names,
        'True_Label': all_true_labels,
        'Predicted_Label': all_pred_labels
    })

    for cls in range(n_classes):
        results_df[f'Prob_Class_{cls}'] = all_pred_probs[:, cls]

    results_excel_path = 'all_samples_predictions.xlsx'
    results_df.to_excel(results_excel_path, index=False)
    print(f"\nAll samples predictions saved to: {results_excel_path}")

    print("\nFinal Results:")
    print(f"Mean AUC: {np.mean(auc_scores_model_4):.4f} ± {np.std(auc_scores_model_4):.4f}")
    print(f"Mean Kappa: {np.mean(kappa_scores):.4f} ± {np.std(kappa_scores):.4f}")

    total_kappa = cohen_kappa_score(all_true_labels, all_pred_labels)
    print(f"Overall Kappa: {total_kappa:.4f}")

    plot_multiclass_roc(all_true_labels, all_pred_probs, n_classes, "roc_curve_overall.pdf")
    print("Overall ROC curve saved as roc_curve_overall.pdf")

    print("\nKappa scores for each fold:")
    for i, kappa in enumerate(kappa_scores, 1):
        print(f"Fold {i}: {kappa:.4f}")