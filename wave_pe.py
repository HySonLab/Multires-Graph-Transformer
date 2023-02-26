import torch
from torch_scatter import scatter_add
import torch_geometric.utils as utils
import numpy as np
from scipy.sparse.linalg import expm

class WaveletPE(object):
    def __init__(self, dim, normalization= "sym", is_undirected = True):
        """
        normalization: for Laplacian None. sym or rw
        """
        ## dimension of the PE
        self.pos_enc_dim = dim
        self.scaling_coefs = [i + 1 for i in range(dim)]
        self.is_undirected = is_undirected
        self.normalization = normalization

    def __call__(self, data):
        data.pos = self.compute_pe(data)
        return data

    def compute_pe(self, graph):
        assert self.pos_enc_dim == len(self.scaling_coefs)
        ### compute normalize graph laplapcian matrix
        edge_index, edge_weight = utils.get_laplacian(
            graph.edge_index,
            graph.edge_weight,
            normalization= self.normalization,
            num_nodes=graph.num_nodes,
        )
        w0 = utils.to_scipy_sparse_matrix(edge_index, edge_weight, graph.num_nodes)
        
        ### scaling parameter s = 1
        vector = torch.zeros((graph.num_nodes, self.pos_enc_dim))
        
        for i in range(self.pos_enc_dim):
            ### multiply the graph laplapcian with the scaling parameter at the current scale.
            w = w0.multiply(-1 * self.scaling_coefs[i])
            w = expm(w)
            
            ## Special case of equivariant encoding where we take the diagonal entries of the wavelet tensor
            vector[:, i] = torch.from_numpy(w.diagonal())
        return vector.float()

def dense_to_sparse(adj):
    edge_index = torch.ones((adj.size(-1), adj.size(-2))).nonzero().t()
    edge_attr = adj[:, edge_index[0], edge_index[1]]
    return edge_index, edge_attr.t()

class Learnable_Equiv_WaveletPE(object):
    def __init__(self, dim, normalization=None, is_undirected = True):
        """
        normalization: for Laplacian None. sym or rw
        """
        ## dimension of the PE
        self.pos_enc_dim = dim
        self.scaling_coefs = [i for i in range(dim)]
        self.is_undirected = is_undirected
        self.normalization = normalization

    def __call__(self, data):
        data.edge_index_wave, data.edge_attr_wave = self.compute_pe(data)
        return data

    def compute_pe(self, graph):
        assert self.pos_enc_dim == len(self.scaling_coefs)
        ### compute normalize graph laplapcian matrix
        edge_index, edge_weight = utils.get_laplacian(
            graph.edge_index,
            graph.edge_weight,
            normalization=self.normalization,
            num_nodes=graph.num_nodes,
        )
        w0 = utils.to_scipy_sparse_matrix(edge_index, edge_weight, graph.num_nodes)
        p = []

        for i in range(self.pos_enc_dim):
            w = w0.multiply(-1 * self.scaling_coefs[i])
            w = expm(w)
            p.append(torch.from_numpy(w.toarray()))
        p = torch.stack(p, dim = 0)
        return dense_to_sparse(p)
    
class EigenDecompose(object):

    '''
        This class is used to compute the eigenvectors and eigenvalues of a Laplacian
        We use these tensors to compute the wavelet tensors which are fed to a learnable equivariant encoding layer.
    
    '''
    def __init__(self, is_undirected = True):
        self.is_undirected = is_undirected

    def __call__(self, data):
        from scipy.sparse.linalg import eigs, eigsh
        eig_fn = eigs if not self.is_undirected else eigsh
        num_nodes = data.num_nodes
        edge_index, edge_weight = utils.get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )
        L = utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        eig_vals, eig_vecs = eig_fn(
            L,
            which='SR' if not self.is_undirected else 'SA',
            return_eigenvectors=True,
        )
        data.EigVals = torch.tensor(eig_vals)
        data.EigVecs = torch.tensor(eig_vecs)
        return data