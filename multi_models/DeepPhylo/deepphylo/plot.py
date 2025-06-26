import matplotlib.pyplot as plt
import numpy as np
# import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
def plot_training(train_losses, val_losses, metrics, title='The twin prediction result: train and test Loss/metrics'):
    def plot_metric(name, metric_values):
        max_metric = max(metric_values)
        plt.plot(metric_values, label=f"{name}")
        # show the digits of max validation aupr in the figure
        plt.plot(metric_values.index(max_metric), max_metric, 'ro')
        # show the digits of max validation aupr in the figure
        plt.annotate(f'{max_metric:.4f}', xy=(metric_values.index(max_metric), max_metric), xytext=(metric_values.index(max_metric), max_metric))
      
    # plot the training and validation loss
    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plot_metric('acc', metrics['acc'])
    plot_metric('mcc', metrics['mcc'])
    plot_metric('roc_auc', metrics['roc_auc'])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss/AUC')
    plt.legend(frameon=False)

def plot_pr_curve(precision, recall, title='Precision-Recall Curve'):
    # 绘制 Precision-Recall 曲线
    plt.figure()
    plt.step(recall, precision, color='black', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # 绘制y=x参考线
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # 计算最接近 y=x 的点
    diff = np.abs(np.array(precision) - np.array(recall))
    min_diff_idx = np.argmin(diff)
    intersect_x = recall[min_diff_idx]
    intersect_y = precision[min_diff_idx]
    
    # 绘制最接近 y=x 的点
    plt.scatter(intersect_x, intersect_y, color='r')
    plt.text(intersect_x, intersect_y, f'({intersect_x:.4f}, {intersect_y:.4f})', 
             verticalalignment='bottom', horizontalalignment='right')
    plt.title(title)
    #plt.legend()
    return intersect_x, intersect_y

def plot_ss_curve(sensitivity, specificity, title='Sensitivity-Specificity Curve'):
    # 绘制 Sensitivity-Specificity 曲线
    plt.figure()
    plt.step(specificity, sensitivity, color='black', alpha=0.2, where='post')
    plt.fill_between(specificity, sensitivity, step='post', alpha=0.2, color='lightskyblue')
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # 绘制y=x参考线, 以及与曲线相交的点
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    # 计算最接近 y=x 的点
    diff = np.abs(np.array(sensitivity) - np.array(specificity))
    min_diff_idx = np.argmin(diff)
    intersect_x = specificity[min_diff_idx]
    intersect_y = sensitivity[min_diff_idx]
    
    # 绘制最接近 y=x 的点
    plt.scatter(intersect_x, intersect_y, color='r', label='Intersection')
    plt.text(intersect_x, intersect_y, f'({intersect_x:.4f}, {intersect_y:.4f})', 
             verticalalignment='bottom', horizontalalignment='right')
    plt.title(title)
    return intersect_x, intersect_y

def plot_age(train_losses, val_losses, val_r2s, title='The simple MLP baseline age prediction result: train and test Loss/R2'):
    # Plot the training and validation
    max_r2 = max(val_r2s)
    # plot the training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.plot(val_r2s, label='Validation R2')
    # show the digits of max validation aupr in the figure
    plt.plot(val_r2s.index(max_r2), max_r2, 'ro')
    # show the digits of max validation aupr in the figure
    plt.annotate(f'{max_r2:.4f}', xy=(val_r2s.index(max_r2), max_r2), xytext=(val_r2s.index(max_r2), max_r2))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss/R2')
    plt.legend(frameon=False)
    plt.show()

def plot_2d(sample_loading,colors, proportion_explained=None):
    fig, ax = plt.subplots()
    if isinstance(sample_loading, pd.DataFrame):
        scatter = ax.scatter(
            sample_loading['PC1'],
            sample_loading['PC2'],
            c=colors,
            cmap='viridis',
            s=8,
            alpha=0.8
        )
        if proportion_explained is not None:
            ax.set_xlabel('PC1: {:.0%}'.format(proportion_explained[0]))
            ax.set_ylabel('PC2: {:.0%}'.format(proportion_explained[1]))
        else:
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
    elif isinstance(sample_loading, np.ndarray):
        scatter = ax.scatter(
            sample_loading[:,0],
            sample_loading[:,1],
            c=colors,
            cmap='viridis',
            s=8,
            alpha=0.8
        )
        if proportion_explained is not None:
            ax.set_xlabel('PC1: {:.0%}'.format(proportion_explained[0]))
            ax.set_ylabel('PC2: {:.0%}'.format(proportion_explained[1]))
        else:
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
    # 创建图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Checherta', markerfacecolor='#1f78b4', markersize=5),
        plt.Line2D([0], [0], marker='o', color='w', label='Iquitos', markerfacecolor='#a6cee3', markersize=5),
        plt.Line2D([0], [0], marker='o', color='w', label='Manaus', markerfacecolor='#ff7f00', markersize=5)
    ]
    ax.legend(handles=legend_elements)
    plt.show()

def normalize_colors(colors):
    normalized_colors = []
    for color in colors:
        normalized_color = tuple(c/255 for c in color)
        normalized_colors.append(normalized_color)
    return normalized_colors

def plot_bar(data_dict, colors=None, rotate=0, label_offset=0, title='Bar Chart', xlabel=None, ylabel='Length',fontsize=15):
    methods = list(data_dict.keys())
    lengths = list(data_dict.values())

    fig, ax = plt.subplots()

    if colors is None:
        colors = 'skyblue'
    elif colors[0][0] > 1:
        colors = normalize_colors(colors)
    ax.bar(methods, lengths, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='x', rotation=rotate)

    for tick in ax.get_xticklabels():
        tick.set_size(fontsize)
        tick.set_ha('right')
        tick.set_rotation(rotate)
        tick.set_position((0,label_offset))

    plt.show()

def plot_box(data_dict, metric='auprc', colors=None, rotate=45, label_offset=0, title='Bar Chart', xlabel=None, ylabel='Length',fontsize=15):
    labels = data_dict.keys()  # 标签
    fig, ax = plt.subplots()
    data = [metric_dict[metric] for metric_dict in data_dict.values()]
    # 绘制箱线图
    boxplot = ax.boxplot(data, patch_artist=True, labels=labels)

    if colors is None:
        colors = 'skyblue'
    elif type(colors[0]) == str:
        colors = [colors]*len(labels)
    elif colors[0][0] > 1:
        colors = normalize_colors(colors)
    # 设置箱线图的颜色
    for patch,color in zip(boxplot['boxes'], colors):
        # r, g, b, a = patch.get_facecolor()
        if type(color) == str:
            patch.set_facecolor(color)
        else:
            patch.set_facecolor((color[0], color[1], color[2], 0.5))

    ax.set_title(title)  # 设置标题
    ax.set_xlabel(xlabel)  # 设置x轴标签
    ax.set_ylabel(ylabel)  # 设置y轴标签

    # 添加散点图
    for (i,dat), color in zip(enumerate(data, 1), colors):
        jitter = np.random.normal(0, 0.04, size=len(dat))
        ax.scatter(np.full(len(dat), i) + jitter, dat, alpha=0.5, color='black', zorder=3)

    for tick in ax.get_xticklabels():
        tick.set_size(fontsize)
        tick.set_ha('right')
        tick.set_rotation(rotate)
        tick.set_position((0,label_offset))
    # # 显示图例
    # ax.legend()

    plt.show()

def summary(array):
    print('mean:', array.mean(), 'std:', array.std(), 'min:', array.min(), 'max:', array.max())

def normalize(data, axis=0):
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    normalized_data = (data - mean) / std
    return normalized_data

def reducer(data_matrix, method='pca', n_components=2, whiten=False):
    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, whiten=whiten)
        embedding = reducer.fit_transform(data_matrix)
        return embedding
    else:
        raise ValueError('Unrecognized method: %s' % method)

    return reducer.fit_transform(data_matrix)

def get_evol_feature(table_matrix_rclr, embeddings):
    evol_feature = []
    for i in range(len(table_matrix_rclr)):
        row = table_matrix_rclr[i]
        idxs = np.where(~np.isnan(row))[0]
        evol_embeds = np.array([embeddings[i] for i in idxs]).sum(axis=0)
        evol_feature.append(evol_embeds)
    return np.array(evol_feature)