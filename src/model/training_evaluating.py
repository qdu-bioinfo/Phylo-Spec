import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

# Evaluate model on test data
def evaluate_model_on_test(model, test_loader, conv_order, data, leaf_to_species, node_weights, num_classes=2):
    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():  # Disable gradient tracking for evaluation
        for inputs, labels in test_loader:
            labels = labels.long()  # Ensure labels are in correct format for loss function
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)

            test_labels.extend(labels.numpy())

            if num_classes == 2:  # Binary classification: use sigmoid
                test_preds.extend(torch.sigmoid(outputs).numpy())
            else:  # Multi-class classification: use softmax
                test_preds.extend(torch.softmax(outputs, dim=1).numpy())

    return np.array(test_labels), np.array(test_preds)

# Calculate ROC AUC score for binary or multi-class classification
def calculate_roc_auc(y_true, y_scores, num_classes):
    if num_classes == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = []
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc.append(auc(fpr, tpr))
    return roc_auc

# Cross-validation training and testing
def cv_train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, conv_order, data, leaf_to_species,
                          node_weights, num_epochs=10, num_classes=2):
    for epoch in range(num_epochs):
        model.train()
        model.clear_accumulated_features()

        for inputs, labels in train_loader:
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                labels = labels.float()
            else:
                labels = labels.long()

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
                if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                    labels = labels.float()
                else:
                    labels = labels.long()

                outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)
                test_group.extend(labels.numpy())

                if num_classes == 2:
                    all_preds.extend(torch.sigmoid(outputs).numpy())
                else:
                    all_preds.extend(torch.softmax(outputs, dim=1).numpy())

    return model, test_group, all_preds

# Standard training loop
def train_model(model, train_loader, criterion, optimizer, conv_order, data, leaf_to_species, node_weights, num_epochs, num_classes):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        model.clear_accumulated_features()

        for inputs, labels in train_loader:
            if num_classes == 2:
                labels = labels.float().unsqueeze(1)  # Binary: shape [batch_size, 1]
            else:
                labels = labels.long()  # Multi-class: integer class indices

            optimizer.zero_grad()
            outputs = model(inputs, conv_order, {}, data, leaf_to_species, labels, node_weights)

            if num_classes == 2:
                loss = criterion(outputs, labels)  # BCEWithLogitsLoss
            else:
                loss = criterion(outputs, labels)  # CrossEntropyLoss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)  # Accumulate total loss

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

    return model
