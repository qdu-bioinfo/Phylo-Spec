import torch
import torch.nn as nn
import re

def convolution_block(in_channels, out_channels, kernel_size=3,padding="same"):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.Dropout(p=0.5),
        nn.ReLU(inplace=False)
    )

class AuxiliaryModel(nn.Module):
    def __init__(self, num_res_blocks=1,channel=16, kernel_size=3):
        super(AuxiliaryModel, self).__init__()
        self.channel = channel
        self.conv1x1_layers = nn.ModuleDict()
        self.res_blocks = nn.ModuleList([convolution_block(self.channel, self.channel, kernel_size=kernel_size) for _ in range(num_res_blocks)])
        self.flatten = nn.Flatten()
    def forward(self, x, conv_order, feature_map, data, leaf_to_species,labels, node_weights):
        self.node_features = {}
        all_features = []
        for leaf, species in leaf_to_species.items():
            node_index = data.columns.get_loc(species) - 1
            feature_map[leaf] = x[:, node_index].view(-1, 1, 1).float()
            layer_name = re.sub(r'\W+', '_', leaf)
            if layer_name not in self.conv1x1_layers:
                self.conv1x1_layers[layer_name] = nn.Conv1d(1, self.channel, kernel_size=1)
            feature_map[leaf] = self.conv1x1_layers[layer_name](feature_map[leaf])
            feature_map[leaf] = feature_map[leaf] * node_weights[leaf]
            all_features.append(feature_map[leaf])
        matched_columns = set(leaf_to_species.values())
        for column in data.columns[1:-1]:
            if column not in matched_columns:
                unmatched_feature = x[:, data.columns.get_loc(column) - 1].view(-1, 1, 1).float()
                layer_name = re.sub(r'\W+', '_', column)
                if layer_name not in self.conv1x1_layers:
                    self.conv1x1_layers[layer_name] = nn.Conv1d(1, self.channel, kernel_size=1)
                unmatched_feature = self.conv1x1_layers[layer_name](unmatched_feature)
                all_features.append(unmatched_feature)
        for children, parent in conv_order:
            child_feats = [feature_map[child] for child in children]
            combined_feat = torch.cat(child_feats, dim=2).float()
            for res_block in self.res_blocks:
                combined_feat = res_block(combined_feat)
            combined_feat = combined_feat * node_weights[parent]
            feature_map[parent] = combined_feat
            all_features.append(feature_map[parent])
        combined_features = torch.cat(all_features, dim=2).float()
        combined_features = nn.MaxPool1d(2)(combined_features)
        combined_features = self.flatten(combined_features)

        return combined_features

class PhyloSpec(nn.Module):
    def __init__(self, fc1_input_dim, num_res_blocks=1, channel=16, kernel_size=3):
        super(PhyloSpec, self).__init__()
        self.channel = channel
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1_layers = nn.ModuleDict()
        self.res_blocks = nn.ModuleList([convolution_block(self.channel, self.channel, kernel_size=kernel_size) for _ in range(num_res_blocks)])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.node_features = {}
        self.accumulated_node_features = {}

    def forward(self, x, conv_order, feature_map, data, leaf_to_species,labels, node_weights):
        all_features = []

        for leaf, species in leaf_to_species.items():
            node_index = data.columns.get_loc(species) - 1
            feature_map[leaf] = x[:, node_index].view(-1, 1, 1).float()
            layer_name = re.sub(r'\W+', '_', leaf)
            if layer_name not in self.conv1x1_layers:
                self.conv1x1_layers[layer_name] = nn.Conv1d(1, self.channel, kernel_size=1)
            feature_map[leaf] = self.conv1x1_layers[layer_name](feature_map[leaf])
            feature_map[leaf] = feature_map[leaf] * node_weights[leaf]
            all_features.append(feature_map[leaf])

            if leaf not in self.accumulated_node_features:
                self.accumulated_node_features[leaf] = []
            self.accumulated_node_features[leaf].append(feature_map[leaf].detach().cpu().numpy())

        matched_columns = set(leaf_to_species.values())
        for column in data.columns[1:-1]:
            if column not in matched_columns:
                unmatched_feature = x[:, data.columns.get_loc(column) - 1].view(-1, 1, 1).float()
                layer_name = re.sub(r'\W+', '_', column)
                if layer_name not in self.conv1x1_layers:
                    self.conv1x1_layers[layer_name] = nn.Conv1d(1, self.channel, kernel_size=1)
                unmatched_feature = self.conv1x1_layers[layer_name](unmatched_feature)
                all_features.append(unmatched_feature)

        for children, parent in conv_order:
            child_feats = [feature_map[child] for child in children]
            combined_feat = torch.cat(child_feats, dim=2).float()
            for res_block in self.res_blocks:
                combined_feat = res_block(combined_feat)
            combined_feat = combined_feat * node_weights[parent]
            feature_map[parent] = combined_feat
            all_features.append(feature_map[parent])

            if parent not in self.accumulated_node_features:
                self.accumulated_node_features[parent] = []
            self.accumulated_node_features[parent].append(combined_feat.detach().cpu().numpy())

        combined_features = torch.cat(all_features, dim=2).float()
        combined_features = nn.MaxPool1d(2)(combined_features)
        combined_features = self.flatten(combined_features)
        root_feat = self.relu(self.fc1(combined_features))
        root_feat = self.relu(self.fc2(root_feat))
        output = self.fc3(root_feat)

        return output

    def clear_accumulated_features(self):
        self.accumulated_node_features = {}

def calculate_fc1_input_dim(aux_model, X, conv_order, data, leaf_to_species, node_weights):
    sample_input = torch.tensor(X, dtype=torch.float32)
    feature_map = {}
    with torch.no_grad():
        combined_features = aux_model(sample_input, conv_order, feature_map, data, leaf_to_species, None,node_weights)
    return combined_features.shape[1]