import torch
from torch_scatter import scatter_add
import torch_geometric.utils as utils
import numpy as np
from scipy.sparse.linalg import expm
    
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform
from torch_geometric.data import Data
import atom3d.util.graph as gr
from torch_geometric.utils import to_dense_adj
from pygsp import graphs, filters
import networkx as nx 
import pymetis

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
        p = []
        for i in range(self.pos_enc_dim):
            ### multiply the graph laplapcian with the scaling parameter at the current scale.
            w = w0.multiply(-1 * self.scaling_coefs[i])
            w = expm(w.todense())
            ## Special case of equivariant encoding where we take the diagonal entries of the wavelet tensor
            p.append(torch.from_numpy(w.diagonal().copy()))
        return torch.stack(p, dim=1)

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
            w = expm(w.todense())
            p.append(torch.from_numpy(w.copy()))
        p = torch.stack(p, dim = 0)
        return dense_to_sparse(p)


def get_wavelet(g, G):
    tensor = []
    for i in range(G.N):
        s = np.zeros(G.N)
        DELTA = i
        s[DELTA] = 1
        s = g.filter(s, method = "chebyshev")
        tensor.append(torch.tensor(s))
    return torch.stack(tensor, axis =0)

def adjacency_matrix_to_list(adjacency_matrix):
    adjacency_list = []
    num_vertices = len(adjacency_matrix)

    for vertex in range(num_vertices):
        adjacent_vertices = []
        for adjacent_vertex in range(num_vertices):
            if adjacency_matrix[vertex][adjacent_vertex] != 0:
                adjacent_vertices.append(adjacent_vertex)
        adjacency_list.append(adjacent_vertices)

    return adjacency_list

class GNNTransformLBA(object):
    def __init__(self, pocket_only=True, num_partition = 5, taus = [10, 20, 30]):
        self.pocket_only = pocket_only
        self.taus = taus
        self.num_partition = num_partition
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        if self.pocket_only:
            item = prot_graph_transform(item, atom_keys=['atoms_pocket'], label_key='scores')
        else:
            item = prot_graph_transform(item, atom_keys=['atoms_protein', 'atoms_pocket'], label_key='scores')
        # transform ligand into PTG graph
        item = mol_graph_transform(item, 'atoms_ligand', 'scores', use_bonds=True, onehot_edges=False)
        node_feats, edges, edge_feats, node_pos = gr.combine_graphs(item['atoms_pocket'], item['atoms_ligand'], edges_between=True)
        combined_graph = Data(node_feats, edges, edge_feats, y=item['scores']['neglog_aff'], pos=node_pos)


        A = to_dense_adj(edges, edge_attr = edge_feats)[0]
        adjacency_list = adjacency_matrix_to_list(A)
        G = nx.from_numpy_matrix(A.numpy())
        n_cuts, membership = pymetis.part_graph(self.num_partition, adjacency=adjacency_list)
        wavelets = torch.zeros((A.shape[0], len(self.taus))).float()

        for i in range(self.num_partition):
            nodes_part_0 = np.argwhere(np.array(membership) == i).ravel()
            subgraph = G.subgraph(nodes_part_0)
            sub_adj = nx.adjacency_matrix(subgraph).todense()
            subgraph = graphs.Graph(sub_adj)
            subgraph.compute_fourier_basis()
            g = filters.Heat(subgraph, self.taus, normalize = True)
            tensor = get_wavelet(g, subgraph)
            tensor = torch.diagonal(tensor).t().float()
            wavelets[nodes_part_0] = tensor
        
        combined_graph.pos = wavelets
        return combined_graph