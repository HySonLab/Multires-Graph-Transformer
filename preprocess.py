import os
from tqdm import tqdm
from network.wave_pe import GNNTransformLBA
import pickle
import warnings 
from atom3d.datasets import LMDBDataset
warnings.filterwarnings("ignore")

def load_data(data_dir, transform = GNNTransformLBA()):
    train_data = LMDBDataset(os.path.join(data_dir, "train"), transform = transform)
    valid_data = LMDBDataset(os.path.join(data_dir, "val"), transform =transform )
    test_data = LMDBDataset(os.path.join(data_dir, 'test'), transform = transform)
    return train_data, valid_data, test_data

data_dir = "data/lba"
transform = GNNTransformLBA(num_partition=10, taus=[10, 15, 20, 25, 30])
train_data, valid_data, test_data = load_data(data_dir, transform = transform)
data = {
    'train' : train_data,
    'val' : valid_data,
    'test' : test_data
}
for type_ in ['train', 'val', 'test']:
    print(f"Processing: {type_} data...")
    graphs = {}
    for i in tqdm(range(len(data[type_]))):
        graphs[i] = data[type_][i]
    file_dir = os.path.join(data_dir, type_)
    with open(os.path.join(file_dir, f"{type_}.pkl"), 'wb') as handle:
        pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done processing \n")