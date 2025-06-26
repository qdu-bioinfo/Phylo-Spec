import math
import torch.nn as nn

class DeepPhylo_classification(nn.Module):
    def __init__(self, hidden_size, embeddings, kernel_size_conv=7, kernel_size_pool=4, dropout_conv=0.2, activation=nn.ReLU()):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=True)
        self.input_size = self.embedding.weight.size(0) - 1
        self.embedding_dim = self.embedding.weight.size(1)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size_conv),
            nn.BatchNorm1d(hidden_size),
            activation,
            nn.MaxPool1d(kernel_size=kernel_size_pool),
        )
        self.fc_abundance = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            activation,
        )
        self.fc_phy = nn.Linear(self.embedding_dim, hidden_size)
        self.fc_pred = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )


    def forward(self, x, ids):
        x_abundance = self.fc_abundance(x)
        embeddings_tensor = self.embedding(ids)
        embeddings_tensor = self.fc_phy(embeddings_tensor)
        embeddings_tensor = embeddings_tensor.transpose(1, 2)
        x_conv = self.conv(embeddings_tensor)
        # x_phy2vec = x_phy2vec.mean(dim=2, keepdim=False)
        x_conv = x_conv.max(dim=2, keepdim=False)[0]
        x = x_abundance * x_conv
        return self.fc_pred(x)

class DeepPhylo_regression(nn.Module):
    def __init__(self, hidden_size, embeddings, kernel_size_conv=7, kernel_size_pool=4, dropout_conv=0.2, activation=nn.ReLU()):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=True)
        self.input_size = self.embedding.weight.size(0) - 1
        self.embedding_dim = self.embedding.weight.size(1)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size_conv),
            nn.BatchNorm1d(hidden_size),
            activation,
            nn.MaxPool1d(kernel_size=kernel_size_pool),
        )
        self.fc_abundance = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            activation,
        )
        self.fc_phy = nn.Linear(self.embedding_dim, hidden_size)
        self.fc_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//8),
            activation,
            nn.Linear(hidden_size//8, 1)
        )


    def forward(self, x, ids):
        x_abundance = self.fc_abundance(x)
        embeddings_tensor = self.embedding(ids)
        embeddings_tensor = self.fc_phy(embeddings_tensor)
        embeddings_tensor = embeddings_tensor.transpose(1, 2)
        x_conv = self.conv(embeddings_tensor)
        # x_phy2vec = x_phy2vec.mean(dim=2, keepdim=False)
        x_conv = x_conv.max(dim=2, keepdim=False)[0]
        x = x_abundance * x_conv
        return self.fc_pred(x)



class DeepPhylo_ibd(nn.Module):
    def __init__(self, hidden_size, embeddings, kernel_size_conv=7, kernel_size_pool=4, dropout_conv=0.2, activation=nn.ReLU()):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=True)
        self.input_size = self.embedding.weight.size(0) - 1
        self.embedding_dim = self.embedding.weight.size(1)
        self.conv = nn.Sequential(
        nn.Conv1d(hidden_size, hidden_size, kernel_size_conv),
            nn.BatchNorm1d(hidden_size),
            activation,
        nn.Dropout(dropout_conv),
            nn.MaxPool1d(kernel_size=kernel_size_pool),
         )
        self.fc_abundance = nn.Sequential(
        nn.Linear(self.input_size, hidden_size),
         activation,
            )
        self.fc_phy = nn.Linear(self.embedding_dim, hidden_size)
        self.fc_pred = nn.Sequential(
        nn.Linear(hidden_size, 1),
            nn.Sigmoid()
         )

    def forward(self, x, ids):
        x_abundance = self.fc_abundance(x)
        embeddings_tensor = self.embedding(ids)
        embeddings_tensor = self.fc_phy(embeddings_tensor)
        embeddings_tensor = embeddings_tensor.transpose(1, 2)
        x_conv = self.conv(embeddings_tensor)
         # x_phy2vec = x_phy2vec.mean(dim=2, keepdim=False)
        x_conv = x_conv.max(dim=2, keepdim=False)[0]
        x = x_abundance * x_conv
        return self.fc_pred(x)



class DeepPhylo_multi_label(nn.Module):
    def __init__(self, hidden_size, embeddings, kernel_size_conv=7, kernel_size_pool=4, dropout_conv=0.2, activation=nn.ReLU()):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=True)
        self.input_size = self.embedding.weight.size(0) - 1
        self.embedding_dim = self.embedding.weight.size(1)
        self.conv = nn.Sequential(
        nn.Conv1d(hidden_size, hidden_size, kernel_size_conv),
            nn.BatchNorm1d(hidden_size),
            activation,
        nn.Dropout(dropout_conv),
            nn.MaxPool1d(kernel_size=kernel_size_pool),
         )
        self.fc_abundance = nn.Sequential(
        nn.Linear(self.input_size, hidden_size),
         activation,
            )
        self.fc_phy = nn.Linear(self.embedding_dim, hidden_size)
        self.fc_pred = nn.Sequential(
        nn.Linear(hidden_size, 4),
            nn.Sigmoid()
         )

    def forward(self, x, ids):
        x_abundance = self.fc_abundance(x)
        embeddings_tensor = self.embedding(ids)
        embeddings_tensor = self.fc_phy(embeddings_tensor)
        embeddings_tensor = embeddings_tensor.transpose(1, 2)
        x_conv = self.conv(embeddings_tensor)
         # x_phy2vec = x_phy2vec.mean(dim=2, keepdim=False)
        x_conv = x_conv.max(dim=2, keepdim=False)[0]
        x = x_abundance * x_conv
        return self.fc_pred(x)