import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

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


def cv_train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, conv_order, data, leaf_to_species,
                           node_weights, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        model.clear_accumulated_features()
        for inputs, labels in train_loader:
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        model.clear_accumulated_features()
        test_group = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                labels = labels.float().unsqueeze(1)
                outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
                test_group.extend(labels.numpy())
                all_preds.extend(torch.sigmoid(outputs).numpy())

    return model, test_group, all_preds

def train_model(model, train_loader, criterion, optimizer, conv_order, data, leaf_to_species, node_weights, num_epochs):
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

    return model