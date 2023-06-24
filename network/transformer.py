import torch
import torch.nn as nn
import torch_geometric.utils as pyg_utils

class TransformerLayer(nn.Module):
    '''
        We re-implement the transformer layer with a modification of the feed forward network 
        to reduce the parameters for fair comparisons with the baselines.
    '''
    def __init__(self, emb_dim, num_head, drop_ratio, norm = "batch"):
        super().__init__()

        self.msa = nn.MultiheadAttention(
             emb_dim, num_head, dropout = drop_ratio, batch_first = True)
        
        if norm == "batch":
            self.norm1 = nn.BatchNorm1d(emb_dim)
            self.norm2 = nn.BatchNorm1d(emb_dim)
        else:
            self.norm1 = nn.LayerNorm(emb_dim)
            self.norm2 = nn.LayerNorm(emb_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.Dropout(drop_ratio),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout(drop_ratio)
        )
        self.norm = norm
        self.dropout = nn.Dropout(drop_ratio)
        self.emb_dim = emb_dim

    def forward(self, h, index = None, is_sparse = True):
        if is_sparse:
            h_dense, mask = pyg_utils.to_dense_batch(h, index)
            h_attn = self.msa(h_dense, h_dense, h_dense,
                             attn_mask = None, key_padding_mask = ~mask)[0][mask]
            h = self.norm1(h + h_attn)
            h = self.norm2(h + self.feed_forward(h))
        else:
            h_dense = h
            h_attn = self.msa(h_dense, h_dense, h_dense)[0]
            Bz, _, _ = h.size()
            if self.norm == "batch":
                h_attn = h_attn.contiguous().view(-1, self.emb_dim)
                h = h.view(-1, self.emb_dim)
            h = self.norm1(h + h_attn)
            h = self.norm2(h + self.feed_forward(h))
            h = h.view(Bz, -1, self.emb_dim)
        h = self.dropout(h)
        return h

class Transformer(nn.Module):
    def __init__(self, emb_dim, num_layer, num_head, drop_ratio, norm = "batch", cat = False):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerLayer(emb_dim, num_head, drop_ratio, norm)
                for i in range(num_layer)
            ]
        )
        self.num_layer = num_layer 

    def forward(self, h, index = None, is_sparse = True):
        for i in range(self.num_layer):
            h = self.layers[i](h, index, is_sparse)
        return h