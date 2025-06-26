import numpy as np
from sklearn.metrics import  accuracy_score, matthews_corrcoef, precision_recall_curve, auc, roc_curve


def select_best_epoch(metrics_dict, metrics_order):
    filtered_epochs = None
    for metric in metrics_order:
        if not metrics_dict[metric]:
            continue
        if len(metrics_dict[metric]) == 1:
            filtered_epochs = metrics_dict[metric][0]
            break
        if filtered_epochs is not None:
            filtered_metric = [(epoch, val) for epoch, val in enumerate(metrics_dict[metric]) if epoch in filtered_epochs]
            max_val = max(filtered_metric, key=lambda x: x[1])
            filtered_epochs = [epoch for epoch, val in filtered_metric if val == max_val[1]]
            if len(filtered_epochs) == 1:
                best_epoch = filtered_epochs[0]
                break
        else:
            max_val = max(enumerate(metrics_dict[metric]), key=lambda x: x[1])
            filtered_epochs = [epoch for epoch, val in enumerate(metrics_dict[metric]) if val == max_val[1]]
            if len(filtered_epochs) == 1:
                best_epoch = filtered_epochs[0]
                break
    return best_epoch

def compute_metrics(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    y_pred_hard = np.where(y_pred > 0.5, 1, 0)
    acc = accuracy_score(y_true, y_pred_hard)
    mcc = matthews_corrcoef(y_true, y_pred_hard)
    fpr, tpr, t = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # 从FPR和TPR计算特异度和敏感度
    specificity = 1 - fpr
    sensitivity = tpr
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    metric_dict = {'acc': acc, 'mcc': mcc, 'roc_auc': roc_auc, 'aupr': aupr, 'precision':precision,'recall':recall, 'specificity':specificity, 'sensitivity':sensitivity}
    return metric_dict

def compute_metrics_ibd(y_true, y_pred):
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    y_pred_hard = np.where(y_pred > 0.5, 1, 0)
    acc = accuracy_score(y_true, y_pred_hard)
    mcc = matthews_corrcoef(y_true, y_pred_hard)
    fpr, tpr, t = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # 从FPR和TPR计算特异度和敏感度
    specificity = 1 - fpr
    sensitivity = tpr
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    # 选取thresholds为0.5时的precision和recall，计算F1
    idx = np.argmin(np.abs(thresholds - 0.5))
    precision = precision[idx]
    recall = recall[idx]
    f1 = 2 * precision * recall / (precision + recall)
    metric_dict = {'acc': acc, 'mcc': mcc, 'roc_auc': roc_auc, 'aupr': aupr, 'f1':f1}
    return metric_dict

def compute_metrics_multi_label(y_true, y_pred):
    metrics_dict = {'acc': [], 'mcc': [], 'roc_auc': [], 'aupr': [], 'f1':[]}
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    for i in range(4):
        y_pred_hard = np.where(y_pred[:,i] > 0.5, 1, 0)
        acc = accuracy_score(y_true[:,i], y_pred_hard)
        mcc = matthews_corrcoef(y_true[:,i], y_pred_hard)
        fpr, tpr, t = roc_curve(y_true[:,i], y_pred[:,i])
        roc_auc = auc(fpr, tpr)
        # 从FPR和TPR计算特异度和敏感度
        specificity = 1 - fpr
        sensitivity = tpr
        precision, recall, thresholds = precision_recall_curve(y_true[:,i], y_pred[:,i])
        aupr = auc(recall, precision)
        # 选取thresholds为0.5时的precision和recall，计算F1
        idx = np.argmin(np.abs(thresholds - 0.5))
        precision = precision[idx]
        recall = recall[idx]
        f1 = 2 * precision * recall / (precision + recall)
        metrics_dict['acc'].append(acc)
        metrics_dict['mcc'].append(mcc)
        metrics_dict['roc_auc'].append(roc_auc)
        metrics_dict['aupr'].append(aupr)
        metrics_dict['f1'].append(f1)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_pred_hard = np.where(y_pred > 0.5, 1, 0)
    acc = accuracy_score(y_true, y_pred_hard)
    mcc = matthews_corrcoef(y_true, y_pred_hard)
    fpr, tpr, t = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # 从FPR和TPR计算特异度和敏感度
    specificity = 1 - fpr
    sensitivity = tpr
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    # 选取thresholds为0.5时的precision和recall，计算F1
    idx = np.argmin(np.abs(thresholds - 0.5))
    precision = precision[idx]
    recall = recall[idx]
    f1 = 2 * precision * recall / (precision + recall)
    metrics_dict['acc'].append(acc)
    metrics_dict['mcc'].append(mcc)
    metrics_dict['roc_auc'].append(roc_auc)
    metrics_dict['aupr'].append(aupr)
    metrics_dict['f1'].append(f1)
    metric_dict2 = {'acc': acc, 'mcc': mcc, 'roc_auc': roc_auc, 'aupr': aupr, 'f1':f1}
    return metrics_dict ,metric_dict2