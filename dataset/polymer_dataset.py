import os
import torch 
import torch_geometric.data as pyg_data
import pandas as pd 
from ogb.utils import mol as mol_utils
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch_geometric.utils as pyg_utils
from rdkit import Chem
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings("ignore")

class PolymerDataset(pyg_data.InMemoryDataset):
    def __init__(self, name, root, type_, transform = None, pre_transform = None, pre_filter = None):
        
        self.name = name 
        self.type_ = type_
        self.root = os.path.join(root, type_)
        print(self.root)
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.path.join(self.root, "raw", f"smiles_{self.type_}.csv")
        
    @property
    def processed_file_names(self):
        if self.pre_transform is not None: 
            return f"wave.pt"
        return f"geometric_{self.name}_unorm.pt" 

    def process(self):
        df = pd.read_csv(self.raw_file_names)
        smiles_list = df["smile"].tolist()

        if self.name == "full":
            target_names = ["gap", "homo", "lumo"]
        else:
            target_names = self.name 
                
        self.size = len(smiles_list)

        data_list = []

        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
            y = df.iloc[i][target_names]
            data_list.append(self.generate_poly_graph(smiles, y))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [data for data in data_list if self.pre_transform(data)]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def generate_poly_graph(self, smi, target):
        mol_graph = mol_utils.smiles2graph(smi)
        x = torch.from_numpy(mol_graph["node_feat"])
        edge_index = torch.from_numpy(mol_graph["edge_index"])
        edge_attr = torch.from_numpy(mol_graph["edge_feat"])
        target = torch.tensor([target]).float()
        data = pyg_data.Data(
            edge_index=edge_index, x=x, edge_attr=edge_attr, y=target)
        return data    
    
    @property
    def smile_list(self):
        df = pd.read_csv(self.raw_file_names)
        return df["smile"].tolist()
