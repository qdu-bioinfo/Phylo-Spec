import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from torch.utils.data import  Dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import random

def set_seed(seed_value=42):

    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    np.random.seed(seed_value)

    random.seed(seed_value)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1234)

def hac(cor):
    def mydist(p1, p2):
        x = int(p1)
        y = int(p2)
        return 1.0 - cor[x, y]
    x = list(range(cor.shape[0]))
    X = np.array(x)
    linked = linkage(np.reshape(X, (len(X), 1)), metric=mydist, method='single')
    result = dendrogram(linked,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True)
    indexes = result.get('ivl')
    del result
    del linked
    index =[]
    for _, itm in enumerate(indexes):
        index.append(int(itm))

    return index

def inverse_C(C):
    return np.log(C) / -4


def reducer(data_matrix, method='pca', n_components=2, whiten=False):
    # if method == 'umap':
    #     reducer = umap.UMAP(n_components=n_components)
    if method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError('Unrecognized method: %s' % method)

    return reducer.fit_transform(data_matrix)

class DeepPhyDataset(Dataset):
    def __init__(self, phy_embedding, data:np.array, label:np.array):
        self.embeddings = torch.cat([torch.zeros(1, phy_embedding.shape[1]), torch.from_numpy(phy_embedding)], dim=0)
        self.X_tensor = torch.from_numpy(data).float()
        self.y_tensor = torch.from_numpy(label).float()

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        nonzero_indices = torch.nonzero(self.X_tensor[idx])
        sample = {
            'X': self.X_tensor[idx],
            'y': self.y_tensor[idx],
            'nonzero_indices': nonzero_indices.squeeze(dim=1) + 1
            }
        return sample
    
    @ staticmethod
    def custom_collate_fn(batch):
        X_batch = [sample['X'] for sample in batch]
        y_batch = [sample['y'] for sample in batch]
        nonzero_indices_batch = [sample['nonzero_indices'] for sample in batch]

        max_seq_len = max([len(indices) for indices in nonzero_indices_batch])

        padded_indices_batch = []
        for indices in nonzero_indices_batch:
            padded_indices = torch.nn.functional.pad(indices, (0, max_seq_len - len(indices)))
            padded_indices_batch.append(padded_indices)

        stacked_X_batch = torch.stack(X_batch)
        stacked_y_batch = torch.stack(y_batch)
        stacked_padded_indices_batch = torch.stack(padded_indices_batch)
        
        assert stacked_X_batch.shape[0] == stacked_y_batch.shape[0] == stacked_padded_indices_batch.shape[0]
        return {'X': stacked_X_batch, 'y': stacked_y_batch, 'nonzero_indices': stacked_padded_indices_batch}