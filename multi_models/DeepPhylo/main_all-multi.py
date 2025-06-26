# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler, label_binarize
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import roc_auc_score, roc_curve, auc, cohen_kappa_score
# import pandas as pd
# import matplotlib.pyplot as plt
# from itertools import cycle
# from 五折.DeepPhylo.deepphylo.pre_dataset import DeepPhyDataset
# from 五折.DeepPhylo.deepphylo.model import DeepPhylo_ibd as DeepPhylo
#
#
# # 设置随机种子，保证结果可复现
# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# # 绘制多分类ROC曲线并保存为PDF
# def plot_multiclass_roc(y_true, y_score, n_classes, filename):
#     # 二值化标签
#     y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
#
#     # 计算每个类别的ROC曲线和AUC
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # 计算微平均ROC曲线和AUC
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     # 绘制ROC曲线
#     plt.figure(figsize=(8, 6))
#     colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(fpr[i], tpr[i], color=color, lw=2,
#                  label='ROC curve of class {0} (AUC = {1:0.2f})'
#                        ''.format(i, roc_auc[i]))
#
#     # 绘制微平均ROC曲线
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (AUC = {0:0.2f})'
#                    ''.format(roc_auc["micro"]),
#              color='deeppink', linestyle=':', linewidth=4)
#
#     plt.plot([0, 1], [0, 1], 'k--', lw=2)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.savefig(filename, format='pdf', bbox_inches='tight')
#     plt.close()
#
#
# # 修改DeepPhylo模型以支持多分类
# class MultiClassDeepPhylo(DeepPhylo):
#     def __init__(self, hidden_size, embeddings, kernel_size_conv=13,
#                  kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU(),
#                  n_classes=2):
#         super().__init__(hidden_size, embeddings, kernel_size_conv,
#                          kernel_size_pool, dropout_conv, activation)
#
#         # 添加维度调整层
#         self.dim_adjust = nn.Linear(1, hidden_size)  # 假设原始输出是1维
#         self.fc = nn.Linear(hidden_size, n_classes)
#
#     def forward(self, X, nonzero_indices):
#         # 获取父类特征
#         features = super().forward(X, nonzero_indices)
#
#         # 调整维度（假设原始输出形状为 [batch_size, 1]）
#         if features.dim() == 3:  # 处理可能的 [batch, channels, 1] 情况
#             features = features.squeeze(-1)
#         features = self.dim_adjust(features)  # 转换为 [batch, hidden_size]
#
#         return self.fc(features)
#
#
# # 模型4的训练部分函数
# def run_model_4_train(X, y, train_idx, val_idx, phy_embedding, n_classes):
#     # 划分训练集和验证集
#     X_train, X_eval = X[train_idx], X[val_idx]
#     y_train, y_eval = y[train_idx], y[val_idx]
#
#     # 使用 SMOTE 进行过采样
#     smote = SMOTE(random_state=42)
#     X_train, y_train = smote.fit_resample(X_train, y_train)
#
#     # 标准化
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_eval = scaler.transform(X_eval)
#
#     # 调用 train 函数进行训练
#     train_losses, val_losses, val_true_labels, val_pred_probs = train(
#         X_train, y_train, X_eval, y_eval, phy_embedding, n_classes)
#
#     # 选择验证集上损失最低的 epoch 作为最佳结果
#     best_epoch = select_best_epoch(val_losses)
#
#     # 获取最佳epoch的预测结果
#     best_true_labels = val_true_labels[best_epoch]
#     best_pred_probs = val_pred_probs[best_epoch]
#
#     # 计算预测类别
#     best_pred_labels = np.argmax(best_pred_probs, axis=1)
#
#     return best_true_labels, best_pred_labels, best_pred_probs
#
#
# # 模型4的 train 函数，包含早停策略
# def train(X_train, Y_train, X_eval, Y_eval, phy_embedding, n_classes,
#           batch_size=32, lr=1e-4, hidden_size=32, kernal_size_conv=13,
#           kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU()):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数进行多分类
#
#     # 构建训练和验证数据集
#     train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               collate_fn=train_dataset.custom_collate_fn)
#     val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                             collate_fn=train_dataset.custom_collate_fn)
#
#     # 初始化修改后的多分类DeepPhylo模型
#     model = MultiClassDeepPhylo(hidden_size=hidden_size,
#                                 embeddings=train_dataset.embeddings,
#                                 kernel_size_conv=kernal_size_conv,
#                                 kernel_size_pool=kernel_size_pool,
#                                 dropout_conv=dropout_conv,
#                                 activation=activation,
#                                 n_classes=n_classes).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#
#     # 提前停止参数
#     epochs = 20
#     patience = 20
#     best_val_loss = float("inf")
#     counter = 0
#
#     # 记录每个 epoch 的训练和验证损失、预测结果
#     train_losses = []
#     val_losses = []
#     val_pred_probs = []  # 存储每个epoch的预测概率
#     val_true_labels = []  # 存储每个epoch的真实标签
#
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         for batch in train_loader:
#             batch = {key: val.to(device) for key, val in batch.items()}
#             optimizer.zero_grad()
#             y_pred_train = model(batch['X'], batch['nonzero_indices'])
#             loss_train = criterion(y_pred_train, batch['y'].long())  # 确保标签是long类型
#             loss_train.backward()
#             optimizer.step()
#             train_loss += loss_train.item() * batch['X'].size(0)
#
#         train_loss /= len(train_loader.dataset)
#         model.eval()
#         val_loss = 0.0
#         epoch_pred_probs = []
#         epoch_true_labels = []
#         with torch.no_grad():
#             for batch in val_loader:
#                 epoch_true_labels.append(batch['y'].numpy())
#                 batch = {key: val.to(device) for key, val in batch.items()}
#                 y_pred_val = model(batch['X'], batch['nonzero_indices'])
#                 loss_val = criterion(y_pred_val, batch['y'].long())
#                 val_loss += loss_val.item() * batch['X'].size(0)
#                 # 使用softmax获取概率
#                 probs = torch.softmax(y_pred_val, dim=1).detach().cpu().numpy()
#                 epoch_pred_probs.append(probs)
#
#         val_loss /= len(val_loader.dataset)
#         epoch_true_labels = np.concatenate(epoch_true_labels)
#         epoch_pred_probs = np.concatenate(epoch_pred_probs, axis=0)
#
#         val_pred_probs.append(epoch_pred_probs)
#         val_true_labels.append(epoch_true_labels)
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#
#         # 提前停止检查
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             counter = 0
#         else:
#             counter += 1
#             if counter >= patience:
#                 break
#
#     del model, optimizer
#     return train_losses, val_losses, val_true_labels, val_pred_probs
#
#
# # 辅助函数：选择验证损失最低的 epoch
# def select_best_epoch(val_losses):
#     best_epoch = np.argmin(val_losses)
#     return best_epoch
#
#
# # 主程序：基于交叉验证训练并评估模型4
# if __name__ == '__main__':
#     set_seed(42)
#
#     # 初始化模型4所需数据文件路径（请根据实际情况修改）
#     X_path = r'/Users/bioinfo/Desktop/after_batch_merged_data_no90%_去0样本3_X.npy'
#     y_path = r'/Users/bioinfo/Desktop/after_batch_merged_data_no90%_去0样本3_y.npy'
#     embedding_path = r'/Users/bioinfo/Desktop/mutl_class_embeding.npy'
#
#     # 加载数据
#     X4 = np.load(X_path, allow_pickle=True)
#     y4 = np.load(y_path, allow_pickle=True)
#     phy_embedding = np.load(embedding_path)
#
#     # 获取类别数量
#     n_classes = len(np.unique(y4))
#     print(f"Number of classes: {n_classes}")
#
#     # 5折交叉验证
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     auc_scores_model_4 = []
#     kappa_scores = []
#
#     # 新增：收集所有预测结果
#     all_true_labels = []
#     all_pred_labels = []
#     all_pred_probs = []
#
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X4, y4)):
#         print(f"\nRunning Fold {fold + 1} for Model 4...")
#         # 确保索引不超出范围
#         train_idx_4 = [i for i in train_idx if i < len(X4)]
#         val_idx_4 = [i for i in val_idx if i < len(X4)]
#
#         # 训练并获取结果
#         true_labels, pred_labels, pred_probs = run_model_4_train(
#             X4, y4, train_idx_4, val_idx_4, phy_embedding, n_classes)
#
#         # 收集结果
#         all_true_labels.extend(true_labels)
#         all_pred_labels.extend(pred_labels)
#         all_pred_probs.append(pred_probs)
#
#         # 计算并存储AUC
#         if n_classes > 2:
#             # 多分类AUC计算
#             y_true_bin = label_binarize(true_labels, classes=np.arange(n_classes))
#             fold_auc = roc_auc_score(y_true_bin, pred_probs, multi_class='ovr')
#             # 绘制并保存ROC曲线
#             plot_filename = f"roc_curve_fold_{fold + 1}.pdf"
#             plot_multiclass_roc(true_labels, pred_probs, n_classes, plot_filename)
#             print(f"ROC curve saved as {plot_filename}")
#         else:
#             # 二分类AUC计算
#             fold_auc = roc_auc_score(true_labels, pred_probs[:, 1])
#
#         auc_scores_model_4.append(fold_auc)
#
#         # 计算并存储Kappa值
#         fold_kappa = cohen_kappa_score(true_labels, pred_labels)
#         kappa_scores.append(fold_kappa)
#
#         print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
#         print(f"Fold {fold + 1} Kappa: {fold_kappa:.4f}")
#
#     # 处理总体预测结果
#     all_pred_probs = np.concatenate(all_pred_probs, axis=0)
#
#     # 输出总体结果
#     print("\nFinal Results:")
#     print(f"Mean AUC: {np.mean(auc_scores_model_4):.4f} ± {np.std(auc_scores_model_4):.4f}")
#     print(f"Mean Kappa: {np.mean(kappa_scores):.4f} ± {np.std(kappa_scores):.4f}")
#
#     # 新增：总体Kappa值
#     total_kappa = cohen_kappa_score(all_true_labels, all_pred_labels)
#     print(f"Overall Kappa: {total_kappa:.4f}")
#
#     # 新增：绘制总体ROC曲线
#     plot_multiclass_roc(all_true_labels, all_pred_probs, n_classes, "roc_curve_overall.pdf")
#     print("Overall ROC curve saved as roc_curve_overall.pdf")
#
#     # 打印每折的Kappa值
#     print("\nKappa scores for each fold:")
#     for i, kappa in enumerate(kappa_scores, 1):
#         print(f"Fold {i}: {kappa:.4f}")


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
from 五折.DeepPhylo.deepphylo.pre_dataset import DeepPhyDataset
from 五折.DeepPhylo.deepphylo.model import DeepPhylo_ibd as DeepPhylo


# 设置随机种子，保证结果可复现
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 绘制多分类ROC曲线并保存为PDF
def plot_multiclass_roc(y_true, y_score, n_classes, filename):
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    # 绘制微平均ROC曲线
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


# 修改DeepPhylo模型以支持多分类
class MultiClassDeepPhylo(DeepPhylo):
    def __init__(self, hidden_size, embeddings, kernel_size_conv=13,
                 kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU(),
                 n_classes=2):
        super().__init__(hidden_size, embeddings, kernel_size_conv,
                         kernel_size_pool, dropout_conv, activation)

        # 添加维度调整层
        self.dim_adjust = nn.Linear(1, hidden_size)  # 假设原始输出是1维
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, X, nonzero_indices):
        # 获取父类特征
        features = super().forward(X, nonzero_indices)

        # 调整维度（假设原始输出形状为 [batch_size, 1]）
        if features.dim() == 3:  # 处理可能的 [batch, channels, 1] 情况
            features = features.squeeze(-1)
        features = self.dim_adjust(features)  # 转换为 [batch, hidden_size]

        return self.fc(features)


# 模型4的训练部分函数
def run_model_4_train(X, y, train_idx, val_idx, phy_embedding, n_classes):
    # 划分训练集和验证集
    X_train, X_eval = X[train_idx], X[val_idx]
    y_train, y_eval = y[train_idx], y[val_idx]

    # 使用 SMOTE 进行过采样
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    # 调用 train 函数进行训练
    train_losses, val_losses, val_true_labels, val_pred_probs = train(
        X_train, y_train, X_eval, y_eval, phy_embedding, n_classes)

    # 选择验证集上损失最低的 epoch 作为最佳结果
    best_epoch = select_best_epoch(val_losses)

    # 获取最佳epoch的预测结果
    best_true_labels = val_true_labels[best_epoch]
    best_pred_probs = val_pred_probs[best_epoch]

    # 计算预测类别
    best_pred_labels = np.argmax(best_pred_probs, axis=1)

    return best_true_labels, best_pred_labels, best_pred_probs


# 模型4的 train 函数，包含早停策略
def train(X_train, Y_train, X_eval, Y_eval, phy_embedding, n_classes,
          batch_size=32, lr=1e-4, hidden_size=32, kernal_size_conv=13,
          kernel_size_pool=4, dropout_conv=0.2, activation=nn.LeakyReLU()):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数进行多分类

    # 构建训练和验证数据集
    train_dataset = DeepPhyDataset(phy_embedding, X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.custom_collate_fn)
    val_dataset = DeepPhyDataset(phy_embedding, X_eval, Y_eval)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=train_dataset.custom_collate_fn)

    # 初始化修改后的多分类DeepPhylo模型
    model = MultiClassDeepPhylo(hidden_size=hidden_size,
                                embeddings=train_dataset.embeddings,
                                kernel_size_conv=kernal_size_conv,
                                kernel_size_pool=kernel_size_pool,
                                dropout_conv=dropout_conv,
                                activation=activation,
                                n_classes=n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 提前停止参数
    epochs = 10
    patience = 20
    best_val_loss = float("inf")
    counter = 0

    # 记录每个 epoch 的训练和验证损失、预测结果
    train_losses = []
    val_losses = []
    val_pred_probs = []  # 存储每个epoch的预测概率
    val_true_labels = []  # 存储每个epoch的真实标签

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()
            y_pred_train = model(batch['X'], batch['nonzero_indices'])
            loss_train = criterion(y_pred_train, batch['y'].long())  # 确保标签是long类型
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
                # 使用softmax获取概率
                probs = torch.softmax(y_pred_val, dim=1).detach().cpu().numpy()
                epoch_pred_probs.append(probs)

        val_loss /= len(val_loader.dataset)
        epoch_true_labels = np.concatenate(epoch_true_labels)
        epoch_pred_probs = np.concatenate(epoch_pred_probs, axis=0)

        val_pred_probs.append(epoch_pred_probs)
        val_true_labels.append(epoch_true_labels)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 提前停止检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    del model, optimizer
    return train_losses, val_losses, val_true_labels, val_pred_probs


# 辅助函数：选择验证损失最低的 epoch
def select_best_epoch(val_losses):
    best_epoch = np.argmin(val_losses)
    return best_epoch


if __name__ == '__main__':
    set_seed(42)

    # 初始化模型4所需数据文件路径（请根据实际情况修改）
    X_path = r'/Users/bioinfo/Desktop/after_batch_merged_data_no90%_去0样本2_X.npy'
    y_path = r'/Users/bioinfo/Desktop/after_batch_merged_data_no90%_去0样本2_y.npy'
    embedding_path = r'/Users/bioinfo/Desktop/mutl_class_embeding.npy'

    # 加载数据
    X4 = np.load(X_path, allow_pickle=True)
    y4 = np.load(y_path, allow_pickle=True)
    phy_embedding = np.load(embedding_path)

    # 获取样本名称（假设样本名称存储在y_path对应的CSV文件中）
    sample_names = pd.read_csv(y_path.replace('.npy', '.csv')).iloc[:, 0].values

    # 获取类别数量
    n_classes = len(np.unique(y4))
    print(f"Number of classes: {n_classes}")

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores_model_4 = []
    kappa_scores = []

    # 新增：收集所有预测结果
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    all_sample_names = []  # 存储所有样本名称

    for fold, (train_idx, val_idx) in enumerate(skf.split(X4, y4)):
        print(f"\nRunning Fold {fold + 1} for Model 4...")
        # 确保索引不超出范围
        train_idx_4 = [i for i in train_idx if i < len(X4)]
        val_idx_4 = [i for i in val_idx if i < len(X4)]

        # 获取当前折的样本名称
        fold_sample_names = sample_names[val_idx_4]

        # 训练并获取结果
        true_labels, pred_labels, pred_probs = run_model_4_train(
            X4, y4, train_idx_4, val_idx_4, phy_embedding, n_classes)

        # 收集结果
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
        all_pred_probs.append(pred_probs)
        all_sample_names.extend(fold_sample_names)

        # 计算并存储AUC
        if n_classes > 2:
            # 多分类AUC计算
            y_true_bin = label_binarize(true_labels, classes=np.arange(n_classes))
            fold_auc = roc_auc_score(y_true_bin, pred_probs, multi_class='ovr')
            # 绘制并保存ROC曲线
            plot_filename = f"roc_curve_fold_{fold + 1}.pdf"
            plot_multiclass_roc(true_labels, pred_probs, n_classes, plot_filename)
            print(f"ROC curve saved as {plot_filename}")
        else:
            # 二分类AUC计算
            fold_auc = roc_auc_score(true_labels, pred_probs[:, 1])

        auc_scores_model_4.append(fold_auc)

        # 计算并存储Kappa值
        fold_kappa = cohen_kappa_score(true_labels, pred_labels)
        kappa_scores.append(fold_kappa)

        print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
        print(f"Fold {fold + 1} Kappa: {fold_kappa:.4f}")

    # 处理总体预测结果
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)

    # 创建包含所有样本预测结果的DataFrame
    results_df = pd.DataFrame({
        'Sample_Name': all_sample_names,
        'True_Label': all_true_labels,
        'Predicted_Label': all_pred_labels
    })

    # 添加每个类别的预测概率
    for cls in range(n_classes):
        results_df[f'Prob_Class_{cls}'] = all_pred_probs[:, cls]

    # 保存到Excel
    results_excel_path = 'all_samples_predictions.xlsx'
    results_df.to_excel(results_excel_path, index=False)
    print(f"\nAll samples predictions saved to: {results_excel_path}")

    # 输出总体结果
    print("\nFinal Results:")
    print(f"Mean AUC: {np.mean(auc_scores_model_4):.4f} ± {np.std(auc_scores_model_4):.4f}")
    print(f"Mean Kappa: {np.mean(kappa_scores):.4f} ± {np.std(kappa_scores):.4f}")

    # 新增：总体Kappa值
    total_kappa = cohen_kappa_score(all_true_labels, all_pred_labels)
    print(f"Overall Kappa: {total_kappa:.4f}")

    # 新增：绘制总体ROC曲线
    plot_multiclass_roc(all_true_labels, all_pred_probs, n_classes, "roc_curve_overall.pdf")
    print("Overall ROC curve saved as roc_curve_overall.pdf")

    # 打印每折的Kappa值
    print("\nKappa scores for each fold:")
    for i, kappa in enumerate(kappa_scores, 1):
        print(f"Fold {i}: {kappa:.4f}")