import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

def evaluate_model_on_test(model, test_loader, conv_order, data, leaf_to_species, node_weights, num_classes=2):
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.long()  # Ensure the labels are in the correct format (long for CrossEntropyLoss)
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
            test_labels.extend(labels.numpy())

            if num_classes == 2:  # Binary classification
                test_preds.extend(torch.sigmoid(outputs).numpy())  # Sigmoid for binary
            else:  # Multi-class classification
                test_preds.extend(torch.softmax(outputs, dim=1).numpy())  # Softmax for multi-class

    return np.array(test_labels), np.array(test_preds)



# def calculate_roc_auc(y_true, y_scores):
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     return roc_auc
def calculate_roc_auc(y_true, y_scores, num_classes):
    if num_classes == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    else:  # Multi-class classification
        # For multi-class, calculate AUC for each class (One-vs-Rest)
        roc_auc = []
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc.append(auc(fpr, tpr))
    return roc_auc


def cv_train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, conv_order, data, leaf_to_species,
                          node_weights, num_epochs=10, num_classes=2):
    for epoch in range(num_epochs):
        model.train()
        model.clear_accumulated_features()

        # 训练阶段
        for inputs, labels in train_loader:
            labels = labels.long()  # 确保标签为 long 类型

            optimizer.zero_grad()
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        model.clear_accumulated_features()
        test_group = []
        all_preds = []

        # 验证阶段
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.long()  # 确保标签为 long 类型
             
                outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
                test_group.extend(labels.numpy())

                if num_classes == 2:  # 二分类
                    all_preds.extend(torch.sigmoid(outputs).numpy())  # Sigmoid for binary
                else:  # 多分类
                    all_preds.extend(torch.softmax(outputs, dim=1).numpy())  # Softmax for multi-class

    return model, test_group, all_preds


def train_model(model, train_loader, criterion, optimizer, conv_order, data, leaf_to_species, node_weights, num_epochs, num_classes):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        model.clear_accumulated_features()

        for inputs, labels in train_loader:
            # 对于二分类，确保标签是 [batch_size, 1]
            if num_classes == 2:
                labels = labels.float().unsqueeze(1)  # 对于二分类，标签应为 [batch_size, 1]
            else:
                labels = labels.long()  # 对于多分类，标签是整数类型

            optimizer.zero_grad()
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)

            # 对于二分类，使用 BCEWithLogitsLoss
            if num_classes == 2:
                loss = criterion(outputs, labels)  # labels 已经是 [batch_size, 1]
            else:
                loss = criterion(outputs, labels)  # 对于多分类，使用 CrossEntropyLoss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

    return model



