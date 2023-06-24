import torch_geometric.utils as pyg_utils
from network.equiv_layers import *


def mask_select(src, dim: int, mask):
    r"""
        Adpated from torch geometric repository
    """
    assert mask.dim() == 1
    assert src.size(dim) == mask.numel()
    dim = dim + src.dim() if dim < 0 else dim
    assert dim >= 0 and dim < src.dim()

    size = [1] * src.dim()
    size[dim] = mask.numel()

    out = src.masked_select(mask.view(size))

    size = list(src.size())
    size[dim] = -1

    return out.view(size)

class EquivWavePE(nn.Module):
    def __init__(self, diff_step, out_dim, device):
        super().__init__()
        self.diff_step = diff_step
        self.out_dim = out_dim
        self.equiv_contractor = layer_2_to_1(diff_step, out_dim, device = device)
        self.norm = nn.BatchNorm1d(out_dim)
    
    def forward(self, batch):
        batch_size = batch.batch[-1] + 1
        adj = pyg_utils.to_dense_adj(batch.edge_index_wave, batch = batch.batch, edge_attr= batch.edge_attr_wave)
        adj = adj.permute(0, 3, 1, 2)
        _, mask = pyg_utils.to_dense_batch(batch.x, batch.batch)
        pos = self.norm(F.relu(self.equiv_contractor(adj)))
        pos = torch.cat([mask_select(pos[i, :, :], 1, mask[i, :]) for i in range(batch_size)], dim = -1)
        return pos.t()