import torch
import torch.nn as nn



class DepGCN(nn.Module):
    """
    Label-aware Dependency Convolutional Neural Network Layer
    """
    def __init__(self, dep_num, dep_dim, in_features, out_features):
        super(DepGCN, self).__init__()
        self.dep_dim = dep_dim
        self.in_features = in_features
        self.out_features = out_features
        self.dep_embedding = nn.Embedding(dep_num, dep_dim, padding_idx=0)
        self.dep_attn = nn.Linear(dep_dim + in_features, out_features)
        self.dep_fc = nn.Linear(dep_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, text, dep_mat, dep_labels):
        dep_label_embed = self.dep_embedding(dep_labels)
        batch_size, seq_len, feat_dim = text.shape
        val_dep = dep_label_embed.unsqueeze(dim=2)
        val_dep = val_dep.repeat(1, 1, seq_len, 1)
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)
        val_sum = torch.cat([val_us, val_dep], dim=-1)
        r = self.dep_attn(val_sum)
        p = torch.sum(r, dim=-1)
        mask = (dep_mat == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)
        output = val_us + self.dep_fc(val_dep)
        output = torch.mul(p_us, output)
        output_sum = torch.sum(output, dim=2)
        output_sum = self.relu(output_sum)
        return output_sum



class ConstGCN(nn.Module):
    """
    Label-aware Constituency Convolutional Neural Network Layer
    """
    def __init__(self, const_num, const_dim, in_features, out_features):
        super(ConstGCN, self).__init__()
        self.const_num = const_num
        self.in_features = in_features
        self.out_features = out_features
        self.const_embedding = nn.Embedding(const_num, const_dim, padding_idx=0)
        self.const_attn = nn.Linear(const_dim + in_features, out_features)
        self.const_fc = nn.Linear(const_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, text, const_mat, const_labels):
        const_label_embed = self.const_embedding(const_labels)
        const_label_embed = torch.mean(const_label_embed, 2)
        batch_size, seq_len, feat_dim = text.shape
        val_dep = const_label_embed.unsqueeze(dim=2)
        val_dep = val_dep.repeat(1, 1, seq_len, 1)
        val_us = text.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, seq_len, 1)
        val_sum = torch.cat([val_us, val_dep], dim=-1)
        r = self.const_attn(val_sum)
        p = torch.sum(r, dim=-1)
        mask = (const_mat == 0).float() * (-1e30)
        p = p + mask
        p = torch.softmax(p, dim=2)
        p_us = p.unsqueeze(3).repeat(1, 1, 1, feat_dim)
        output = val_us + self.const_fc(val_dep)
        output = torch.mul(p_us, output)
        output_sum = torch.sum(output, dim=2)
        output_sum = self.relu(output_sum)
        return output_sum
