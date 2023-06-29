import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn 
import torch_geometric.utils as pyg_utils
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from network.transformer import *
from network.conv import *
from network.gps_layer import GPSLayer
from network.learnable_equiv_wave import EquivWavePE

class GraphPooling(nn.Module):
    def __init__(self, emb_dim, out_dim, dropout, norm = "layer", lin = True, gnn_type = "gine"):
        super().__init__()
        if gnn_type == "gine":
            self.conv1 = pyg_nn.GINEConv(nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.BatchNorm1d(emb_dim*2) if norm == "batch" else nn.LayerNorm(emb_dim * 2)
                                                    , nn.ReLU(), nn.Linear(emb_dim * 2, emb_dim)), edge_dim = emb_dim)
            self.conv2 = pyg_nn.GINEConv(nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.BatchNorm1d(emb_dim * 2) if norm == "batch" else nn.LayerNorm(emb_dim * 2)
                                                    , nn.ReLU(), nn.Linear(emb_dim * 2, out_dim)), edge_dim = emb_dim)
        elif gnn_type == "graphconv":
            self.conv1 = pyg_nn.GraphConv(emb_dim, emb_dim)
            self.conv2 = pyg_nn.GraphConv(emb_dim, emb_dim)
        elif gnn_type == "CustomGatedGCN":
            self.conv1 = GatedGCNLayer(emb_dim, emb_dim, 0, False, True)
            self.conv2 = GatedGCNLayer(emb_dim, out_dim, 0, False, True)

        self.gnn_type = gnn_type
        self.norm1 = nn.LayerNorm(emb_dim) if norm == "layer" else nn.BatchNorm1d(emb_dim)
        self.norm2 = nn.LayerNorm(out_dim) if norm == "layer" else nn.BatchNorm1d(out_dim)
        
        if lin:
            self.lin = nn.Linear(emb_dim + out_dim, out_dim)
        else:
            self.lin = None 

    def forward(self, x, edge_index, edge_attr, pos):
        h0 = x
        if self.gnn_type == "CustomGatedGCN":
            h1, e1 = self.conv1(h0, edge_index, edge_attr, pos)
            h1 = self.norm1(F.relu(h1))
            h2, e2 = self.conv2(h1, edge_index, e1, pos)
            h2 = self.norm2(F.relu(h2))
        elif self.gnn_type == "gine":
            h1 = self.conv1(h0, edge_index, edge_attr) 
            h1 = self.norm1(F.relu(h1))
            h2 = self.conv2(h1, edge_index, edge_attr) 
            h2 = self.norm2(F.relu(h2))
        elif self.gnn_type == "graphconv":
            h1 = self.conv1(h0, edge_index)
            h1 = self.norm1(F.relu(h1))
            h2 = self.conv2(h1, edge_index)
            h2 = self.norm2(F.relu(h2)) 
        h = torch.cat([h1, h2], dim = -1)
        h = F.relu(self.lin(h))
        return h

class ClusterLearner(nn.Module):
    def __init__(self, gnn_type, emb_dim, dropout,norm, num_cluster):
        super().__init__()
        self.gnn_pool = GraphPooling(emb_dim, num_cluster, dropout, norm, gnn_type = gnn_type)
        self.gnn_emb = GraphPooling(emb_dim, emb_dim, dropout, norm, gnn_type = gnn_type)

        self.gnn_type = gnn_type

    def forward(self, x, edge_index, edge_attr, pos, batch):
        s = self.gnn_pool(x, edge_index, edge_attr, pos)
        z = self.gnn_emb(x, edge_index, edge_attr, pos)
        dense_z, mask1 = pyg_utils.to_dense_batch(z, batch = batch)
        dense_s, mask2 = pyg_utils.to_dense_batch(s, batch = batch)
        adj = pyg_utils.to_dense_adj(edge_index, batch = batch)    
        out, out_adj, link_loss, ent_loss = pyg_nn.dense_diff_pool(dense_z, adj, dense_s, mask1)
        return out, out_adj, link_loss, ent_loss, s, mask1

class MLP_GraphPooling(nn.Module):
    def __init__(self, emb_dim, out_dim, dropout, norm = "layer_norm", lin = True, gnn_type = "gine"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), 
            nn.BatchNorm1d(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.BatchNorm1d(emb_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class MLP_ClusterLearner(nn.Module):
    def __init__(self, emb_dim, dropout, norm, num_cluster):
        super().__init__()
        self.x_pool = MLP_GraphPooling(emb_dim, emb_dim, dropout, norm)
        self.a_pool = MLP_GraphPooling(emb_dim, num_cluster, dropout, norm)
    
    def forward(self, x, edge_index, batch):
        s = self.a_pool(x)
        z = self.x_pool(x)
        dense_z, mask1 = pyg_utils.to_dense_batch(z, batch = batch)
        dense_s, mask2 = pyg_utils.to_dense_batch(s, batch = batch)
        adj = pyg_utils.to_dense_adj(edge_index, batch = batch)    
        out, out_adj, link_loss, ent_loss = pyg_nn.dense_diff_pool(dense_z, adj, dense_s, mask1)
        return out, out_adj, link_loss, ent_loss, s, mask1

class MGT(nn.Module):
    def __init__(self, num_layer, emb_dim, pos_dim, num_task, num_head, dropout, attn_dropout, norm, num_cluster, gnn_type, pe_name, device):
        super().__init__()

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        if pe_name == "learnable_equiv_wave":
            self.pos_encoder = EquivWavePE(pos_dim, 16, device)
        else:
            self.pre_norm = nn.LayerNorm(pos_dim) if norm == "layer_norm" else nn.BatchNorm1d(pos_dim)
            self.pos_encoder = nn.Linear(pos_dim, 16)

        self.lin_pos = nn.Linear(emb_dim + 16, emb_dim)
        self.pe_name = pe_name

        self.gps = nn.ModuleList([])

        for i in range(num_layer):
            if gnn_type == "graphconv":
                self.gps.append(pyg_nn.GPSConv(emb_dim, pyg_nn.GraphConv(emb_dim, emb_dim), 
                                heads = num_head, dropout = dropout, attn_dropout=attn_dropout, norm = norm))
            elif gnn_type == "gine":
                self.gps.append(pyg_nn.GPSConv(emb_dim, pyg_nn.GINEConv(nn.Sequential(nn.Linear(emb_dim, emb_dim * 2), nn.BatchNorm1d(emb_dim * 2) if norm == "batch_norm" else nn.LayerNorm(emb_dim * 2)
                                                                   ,nn.ReLU(), nn.Linear(emb_dim * 2, emb_dim)), 
                                                                   edge_dim = emb_dim), heads = num_head, dropout = dropout, 
                                                                   attn_dropout = attn_dropout, norm = norm + "_norm"))
            else:
                raise NotImplementedError
        
        self.gnn_type = gnn_type
        self.num_layer = num_layer

        self.cluster_learner = ClusterLearner(gnn_type, emb_dim, dropout, norm, num_cluster)

        encoder_layer = nn.TransformerEncoderLayer(emb_dim, num_head, dim_feedforward = emb_dim * 2, dropout= dropout, activation='relu', layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None)
        self.substructure_transformer = nn.TransformerEncoder(encoder_layer, 4)
        
        self.lin = nn.Linear(emb_dim * 2, emb_dim)

        self.ffn = nn.Linear(emb_dim, num_task)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def forward(self, batch):
        batch.x = self.atom_encoder(batch.x)
        batch.edge_attr = self.bond_encoder(batch.edge_attr)
        
        if self.pe_name == "learnable_equiv_wave":
            batch.pos = self.pos_encoder(batch)
        else:
            batch.pos = self.pos_encoder(self.pre_norm(batch.pos))

        batch.x = torch.cat([batch.x, batch.pos], dim = -1)
        batch.x = self.lin_pos(batch.x)

        for i in range(self.num_layer):
            if self.gnn_type == "gine":
                batch.x = self.gps[i](batch.x, batch.edge_index, batch.batch, edge_attr = batch.edge_attr)
            else:
                batch.x = self.gps[i](batch.x, batch.edge_index, batch.batch)
        
        h_units, _, loss1, loss2, s, mask = self.cluster_learner(batch.x, batch.edge_index, batch.edge_attr, batch.pos, batch.batch)
        
        h_transform = self.substructure_transformer(h_units)
        h_units = torch.cat([h_transform, h_units], dim = -1)
        h_units = self.lin(h_units)
        
        h_g = F.relu(torch.sum(h_units, dim = 1))
        return self.ffn(h_g), loss1, loss2

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CustomMGT(nn.Module):
    """"
        Customize MGT to be adapted with GatedGCN Layer
    """
    def __init__(self, config):
        super().__init__()
        print(config.norm)

        ##### atom and bond embdding
        self.emb_dim = config.emb_dim
        self.atom_encoder = AtomEncoder(config.emb_dim)
        self.bond_encoder = BondEncoder(config.emb_dim)
       

        ##### positional encoding #####
        if config.pe_name == "learnable_equiv_wave":
            self.pos_encoder = EquivWavePE(config.pos_dim, 16, config.device)
        else:
            self.pre_norm = nn.LayerNorm(config.pos_dim) if config.norm == "layer" else nn.BatchNorm1d(config.pos_dim)
            self.pos_encoder = nn.Linear(config.pos_dim, 16)

        self.lin_pos = nn.Linear(config.emb_dim + 16, config.emb_dim)
        self.pe_name = config.pe_name

        ##### atom-level transformer ##### 
    
        Layer = GPSLayer

        layers = []
        for _ in range(config.num_layer):
            layers.append(Layer(dim_h=config.emb_dim,
                                local_gnn_type= config.local_gnn_type,
                                global_model_type= config.global_model_type,
                                num_heads=config.num_head,
                                equivstable_pe= True,
                                dropout = config.dropout,
                                attn_dropout=config.attn_dropout,
                                layer_norm = config.norm == "layer",
                                batch_norm = config.norm == "batch",
                                bigbird_cfg=None))
        
        self.atom_transformer = torch.nn.Sequential(*layers)
       
        ##### clustering #####
        self.cluster_learner = ClusterLearner("CustomGatedGCN", config.emb_dim, config.dropout, config.norm, config.num_cluster)

        ##### substructure-level transformer #####
        norm = "layer" if config.norm == "layer" else "batch"
        self.substructure_transformer = Transformer(emb_dim = config.emb_dim, 
                                                    num_layer = 2, 
                                                    num_head = config.num_head,
                                                    drop_ratio = config.dropout, 
                                                    norm = norm)
        self.lin = nn.Linear(config.emb_dim * 2, config.emb_dim)


        ##### feed-forward nn layer #####
        self.ffn = nn.Linear(config.emb_dim, config.num_task)

    def forward(self, batch):
        ### atom and bond encoder ####
        batch.x = self.atom_encoder(batch.x)
        batch.edge_attr = self.bond_encoder(batch.edge_attr)

        ### positional encoding ####
        if self.pe_name == "learnable_equiv_wave":
            batch.pos = self.pos_encoder(batch)
        else:
            batch.pos = self.pos_encoder(self.pre_norm(batch.pos))
        batch.x = torch.cat([batch.x, batch.pos], dim = 1)
        batch.x = self.lin_pos(batch.x)

        ### atom-level transformer ###
        h = batch.x
        batch = self.atom_transformer(batch)

        ### clustering ###
        h_atom = batch.x 
        h_pos = batch.pos
        h_units, _, loss1, loss2, s, mask = self.cluster_learner(h_atom, batch.edge_index, 
                                                                batch.edge_attr, h_pos, batch.batch)

        ### substructure-level transformer
        h_transform = self.substructure_transformer(h_units, is_sparse = False)
        h_units = torch.cat([h_transform, h_units], dim = -1)
        h_units = self.lin(h_units)
        
        h_g = torch.sum(h_units, dim = 1)
        return self.ffn(h_g), loss1, loss2
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LBAMGT(nn.Module):
    def __init__(self, config):
        super().__init__()

        ##### atom and bond embdding
        self.emb_dim = config.emb_dim

        self.atom_encoder = nn.Sequential(
                nn.Linear(61, config.emb_dim),
                nn.Tanh(),
                nn.Linear(config.emb_dim, config.emb_dim)
                )
        
        self.bond_encoder = nn.Sequential(
                nn.Linear(1, config.emb_dim), 
                nn.Tanh(),
                nn.Linear(config.emb_dim, config.emb_dim)
                )
            #self.bond_encoder = nn.Embedding(4, config.emb_dim)

        ##### positional encoding #####
        self.pe_name = config.pe_name
        num_pos =config.diff_step
        self.equiv_pe = config.equiv_pe
        if config.equiv_pe:
            print("Used Equiv PE\n")
            self.pos_encoder = EquivWavePE(num_pos, 16, config.device)
        else:
            self.pre_norm = nn.BatchNorm1d(num_pos)
            self.pos_encoder = nn.Linear(num_pos, 16)
        

        self.lin1 = nn.Linear(config.emb_dim + 16, config.emb_dim)


        ##### atom-level transformer ##### 
        Layer = GPSLayer
        layers = []
        for _ in range(config.num_layer):
            layers.append(Layer(dim_h=config.emb_dim,
                                local_gnn_type=config.local_gnn_type,
                                global_model_type= config.global_model_type,
                                num_heads=config.num_head,
                                equivstable_pe=config.local_gnn_type == "CustomGatedGCN",
                                dropout = config.dropout,
                                attn_dropout=config.attn_dropout,
                                layer_norm = config.layer_norm,
                                batch_norm = config.batch_norm,
                                bigbird_cfg=None))
        
        self.atom_transformer = torch.nn.Sequential(*layers)
        
        ##### clustering #####
        self.cluster_learner = ClusterLearner(config.local_gnn_type, config.emb_dim, config.dropout, config.norm, config.num_cluster)

        ##### substructure-level transformer #####
        norm = "layer" if config.layer_norm == 1 else "batch"
        self.substructure_transformer = Transformer(emb_dim = config.emb_dim, 
                                                    num_layer = 2, 
                                                    num_head = config.num_head,
                                                    drop_ratio = config.dropout, 
                                                    norm = norm)
        self.lin = nn.Linear(config.emb_dim * 2, config.emb_dim)

        ##### feed-forward nn layer #####
        self.ffn = nn.Linear(config.emb_dim, config.num_task)

    def forward(self, batch, return_feature = False):
        ### atom and bond encoder ####
        #print(batch.pos.shape)
        batch.x = self.atom_encoder(batch.x)
        batch.edge_attr = self.bond_encoder(batch.edge_attr.unsqueeze(-1))

        ### positional encoding ####    
        if self.equiv_pe:
            ## newly added code
            batch.pos = self.pos_encoder(batch)
            #batch.pos = self.lin_pos(batch.pos)
        else:
            batch.pos = self.pos_encoder(self.pre_norm(batch.pos))
        batch.x = torch.cat([batch.x, batch.pos], dim = 1)
        batch.x = self.lin1(batch.x)

        ### atom-level transformer ###
        h = batch.x
        batch = self.atom_transformer(batch)

        ### clustering ###
        h_atom = batch.x 
        h_pos = batch.pos
        h_units, _, loss1, loss2, s, mask = self.cluster_learner(h_atom, batch.edge_index, 
                                                                batch.edge_attr, h_pos, batch.batch)



        # ### substructure-level transformer
        h_transform = self.substructure_transformer(h_units, is_sparse = False)
        h_units = torch.cat([h_transform, h_units], dim = -1)
        h_units = self.lin(h_units)
        h_g = torch.sum(h_units, dim = 1)

        if return_feature:
            return h_g
        return self.ffn(h_g), loss1, loss2, s

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
