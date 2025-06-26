import pickle
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

def evaluate_model_on_test(model, test_loader, conv_order, data, leaf_to_species, node_weights):
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.float().unsqueeze(1)
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
            test_labels.extend(labels.numpy())
            test_preds.extend(torch.sigmoid(outputs).numpy())
    return np.array(test_labels), np.array(test_preds)

def calculate_roc_auc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

import torch
import numpy as np

def cv_train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, conv_order, data, leaf_to_species,
                          num_classes, node_weights, num_epochs=15):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        model.clear_accumulated_features()

        for inputs, labels in train_loader:
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        model.clear_accumulated_features()
        running_loss = 0.0
        test_group = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.float().unsqueeze(1)
                outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                test_group.extend(labels.numpy())
                all_preds.extend(torch.sigmoid(outputs).numpy())

        val_loss = running_loss / len(test_loader.dataset)
        val_losses.append(val_loss)

        y_val_pred = (np.array(all_preds) >= 0.5).astype(int)
        accuracy, precision, recall, f1, kappa = calculate_metrics(np.array(test_group), y_val_pred)

        roc_auc = calculate_roc_auc(np.array(test_group), np.array(all_preds))

    return model, test_group, all_preds, train_losses, val_losses

def train_model(model, train_loader, criterion, optimizer, conv_order, data, leaf_to_species, node_weights, num_epochs=15):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        model.clear_accumulated_features()

        for inputs, labels in train_loader:
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

    torch.save(model.state_dict(), 'final_trained_model.pth')

    return train_losses, model

def calculate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    kappa = cohen_kappa_score(y_true, y_pred_binary)
    return accuracy, precision, recall, f1, kappa