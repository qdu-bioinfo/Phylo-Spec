import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve, auc, precision_recall_curve
import random
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


random.seed(42)
np.random.seed(42)

# output folde
output_folder = r'output_folder'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# input
csv_path = r'csv'
data = pd.read_csv(csv_path)


sample_names = data.iloc[:, 0].values
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y)
class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(f"Class mapping: {class_mapping}")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_kappa = []
fold_auc = []
fold_pr_auc = []
all_true = []
all_pred = []
all_probas = []
all_sample_names = []


for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    sample_names_fold = sample_names[test_index]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=500, max_depth=4, max_features="log2", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    all_true.extend(y_test)
    all_pred.extend(y_pred)
    all_probas.append(y_pred_prob)
    all_sample_names.extend(sample_names_fold)

    kappa = cohen_kappa_score(y_test, y_pred)
    fold_kappa.append(kappa)

    auc_score = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    fold_auc.append(auc_score)

    pr_auc = 0
    for i in range(len(np.unique(y))):
        precision, recall, _ = precision_recall_curve(y_test == i, y_pred_prob[:, i])
        pr_auc += auc(recall, precision)
    fold_pr_auc.append(pr_auc / len(np.unique(y)))

    print(f"Fold {fold_idx} - Kappa: {kappa:.4f}, AUC: {auc_score:.4f}, PR AUC: {pr_auc / len(np.unique(y)):.4f}")

overall_true = np.array(all_true)
overall_pred = np.array(all_pred)
overall_probas = np.concatenate(all_probas, axis=0)

overall_kappa = cohen_kappa_score(overall_true, overall_pred, weights='quadratic')
overall_auc = roc_auc_score(overall_true, overall_probas, multi_class='ovr')

print(f"\nOverall Metrics:")
print(f"Kappa: {overall_kappa:.4f}")
print(f"AUC: {overall_auc:.4f}")
print(f"Avg Fold Kappa: {np.mean(fold_kappa):.4f}")
print(f"Avg Fold PR AUC: {np.mean(fold_pr_auc):.4f}")

############################################
# ROC
############################################
def plot_multiclass_roc(y_true, y_score, n_classes, pdf_filename):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(pdf_filename, format='pdf')
    plt.close()


roc_pdf_path = os.path.join(output_folder, 'multiclass_roc.pdf')
plot_multiclass_roc(overall_true, overall_probas, len(le.classes_), roc_pdf_path)
print(f"ROC curve saved to: {roc_pdf_path}")

