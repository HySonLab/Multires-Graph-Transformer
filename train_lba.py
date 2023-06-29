import argparse
import torch_geometric
from torch_geometric.loader import DataLoader
import pickle
import os 
from utils import *
from tqdm import tqdm
from metrics import *
from network.model import LBAMGT

parser = argparse.ArgumentParser(description="MGT on Protein-Ligand Affinity Prediction")
parser.add_argument("--out_dir", type = str, default = "./results")
parser.add_argument("--seed", type = int, default=123456)
parser.add_argument("--pe_name", type = str, default = "wave")
parser.add_argument("--device", type=str, default="cuda:0",help = "cuda device")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--num_layer", type = int, default = 2)
parser.add_argument("--num_epoch", type = int, default = 200)
parser.add_argument("--num_head", type = int, default = 4)
parser.add_argument("--norm", type = str, default = "batch")
parser.add_argument('--emb_dim', type = int, default = 100)
parser.add_argument("--num_task", type = int, default = 11)
parser.add_argument("--dropout", type = float, default=0.25)
parser.add_argument("--residual", type = int, default = 1)
parser.add_argument("--layer_norm", type = int, default=0)
parser.add_argument("--batch_norm", type = int, default = 1)
parser.add_argument("--num_cluster", type = int, default = 10)
parser.add_argument("--attn_dropout", type = float, default = 0.25)
parser.add_argument("--local_gnn_type", type = str, default = "CustomGatedGCN")
parser.add_argument("--global_model_type", type = str, default = "Transformer")
parser.add_argument("--data_format", type = str, default = "ogb")
parser.add_argument("--diff_step", type = int, default = 5)

args = parser.parse_args()
torch_geometric.seed_everything(args.seed)


filedir = "data/lba"


with open(os.path.join(filedir, "train/train.pkl"), 'rb') as handle:
    b = pickle.load(handle)
    train_data = list(b.values())

with open(os.path.join(filedir, "val/val.pkl"), "rb") as handle:
    b = pickle.load(handle)
    val_data = list(b.values())

with open(os.path.join(filedir, "test/test.pkl"), "rb") as handle:
    b = pickle.load(handle)
    test_data = list(b.values())


print("Num train: ", len(train_data))
print("Num valid: ", len(val_data))
print("Num test: ", len(test_data))

batch_size = args.batch_size

train_loader = DataLoader(train_data, batch_size = 32, shuffle = True, num_workers = 6)
val_loader = DataLoader(val_data, batch_size = 32, shuffle = False, num_workers = 6)
test_loader = DataLoader(test_data, batch_size = 32, shuffle = False, num_workers = 6)


device = args.device
print("Device: ", device)
num_layer = args.num_layer
num_head = args.num_head
args.equiv_pe=0

args.residual = bool(args.residual)
args.layer_norm = bool(args.layer_norm)
args.batch_norm = bool(args.batch_norm)


model = LBAMGT(args).to(device)

###

print("Number of parameters: ", model.num_parameters())

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr = 1e-5, factor=0.5)

model.train()

evaluator = MetricWrapper("mse")
best_val_rmse = 1e9
test_rmse_at_best_val_rmse = -1

num_epoch = args.num_epoch 
for epoch in range(1, num_epoch + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    train_mae = train_with_cluster(model, device, train_loader, optimizer, [-1], is_lba = True)
    

    print('Evaluating...')
    val_mse = eval_with_cluster(model, device, val_loader, evaluator, is_lba = True)
    test_mse = eval_with_cluster(model, device, test_loader, evaluator, is_lba = True)

    valid_rmse = val_mse ** 0.5
    test_rmse = test_mse ** 0.5
    
    scheduler.step(val_mse)

    if best_val_rmse > valid_rmse:
        best_val_rmse = valid_rmse
        test_rmse_at_best_val_rmse = test_rmse 
 
    print('| Train Loss: {:5.5f} | Validation RMSE: {:5.5f} | Test RMSE: {:5.5f}'.format(train_mae, valid_rmse, test_rmse))

print()
print("Seed: ", args.seed)
print("Number of parameters: ", model.num_parameters())
print("Local GNN: ", args.local_gnn_type)
print("Global model: ", args.global_model_type)
print("Num layer: ", args.num_layer)
print()
print("Best validation RMSE: ", best_val_rmse)
print("Test RMSE: ", test_rmse_at_best_val_rmse)