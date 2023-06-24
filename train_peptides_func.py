import argparse
import os 
import torch_geometric
from torch_geometric.loader import DataLoader
import pickle
from torch_geometric.transforms import AddRandomWalkPE, AddLaplacianEigenvectorPE

from utils import *
from metrics import eval_ap
from network.model import *
from dataset.peptides_dataset import PeptidesFunctionalDataset
from network.wave_pe import WaveletPE, Learnable_Equiv_WaveletPE

parser = argparse.ArgumentParser(description="MGT on Peptides Functional Prediction")
parser.add_argument("--out_dir", type = str, default = "./results")
parser.add_argument("--seed", type = int, default=123456)
parser.add_argument("--pe_name", type = str, default = "spectral")
parser.add_argument("--device", type=str, default="cuda:0", help = "cuda device")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--num_layer", type = int, default = 2)
parser.add_argument("--num_epoch", type = int, default = 200)
parser.add_argument("--num_head", type = int, default = 4)
parser.add_argument("--norm", type = str, default = "batch")
parser.add_argument('--emb_dim', type = int, default = 84)
parser.add_argument("--num_task", type = int, default = 10)
parser.add_argument("--dropout", type = float, default=0.25)
parser.add_argument("--residual", type = int, default = 1)
parser.add_argument("--num_cluster", type = int, default = 10)
parser.add_argument("--attn_dropout", type = float, default = 0.5)
parser.add_argument("--local_gnn_type", type = str, default = "CustomGatedGCN")
parser.add_argument("--global_model_type", type = str, default = "Transformer")
parser.add_argument("--pos_dim", type = int, default = 8)
parser.add_argument("--version", type = str, default = "custom")
parser.add_argument("--gnn_type", type = str, default = "gine")

args = parser.parse_args()
torch_geometric.seed_everything(1235)

if args.pe_name == "wave":
    ### Wavelet should be preprocessed to reduce loading time in dataloader during the optimization phase ###
    pre_transform = WaveletPE(args.pos_dim, is_undirected = True)
    dataset = PeptidesFunctionalDataset(pre_transform=pre_transform)
if args.pe_name == "learnable_equiv_wave":
    pre_transform = Learnable_Equiv_WaveletPE(args.pos_dim, is_undirected=True)
    dataset = PeptidesFunctionalDataset(pre_transform = pre_transform)
elif args.pe_name == "laplacian":
    transform = AddLaplacianEigenvectorPE(args.pos_dim, attr_name = "pos", is_undirected=True)
    dataset = PeptidesFunctionalDataset(transform = transform)
elif args.pe_name == "random_walk":
    transform = AddRandomWalkPE(args.pos_dim, attr_name = "pos")
    dataset = PeptidesFunctionalDataset(transform=transform)
else:
    raise NotImplementedError


split = dataset.get_idx_split()
train_data = dataset[split['train']]
valid_data = dataset[split["val"]]
test_data = dataset[split["test"]]

print("Num train: ", len(train_data))
print("Num valid: ", len(valid_data))
print("Num test: ", len(test_data))


train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = 6, pin_memory = False)
valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, num_workers = 6, pin_memory = False)
test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = 6, pin_memory = False)


device = args.device
num_layer = args.num_layer
num_head = args.num_head


for batch in train_loader:
    print(batch)
    break

if args.version == "custom":
    model = CustomMGT(args).to(args.device)
else:
    model = MGT(args.num_layer, args.emb_dim, args.pos_dim, args.num_task, args.num_head, args.dropout, 
                args.attn_dropout, args.norm, args.num_cluster, args.gnn_type, args.pe_name, args.device).to(args.device)

print("Number of parameters: ", model.num_parameters)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', min_lr = 1e-6, verbose=True, factor = 0.5)
evaluator = eval_ap
best_val_loss = 1e9
best_val_ap = 0
test_ap_at_best_val_ap = -1
valid_loss = None

 
for epoch in range(1, args.num_epoch + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    train_loss = train_with_cluster(model, args.device, train_loader, optimizer, ["classification"])

    print('Evaluating...')
    valid_ap = eval_with_cluster(model, args.device, valid_loader, evaluator, "classification")
    test_ap = eval_with_cluster(model, args.device, test_loader, evaluator, "classification")
    
    scheduler.step(valid_ap)
    if best_val_ap < valid_ap:
        best_val_ap = valid_ap
        test_ap_at_best_val_ap = test_ap

    print('| Train Loss: {:5.5f} | Validation AP: {:5.5f} | Test AP: {:5.5f}'.format(train_loss, valid_ap, test_ap))

print()
print()
print(args)
print()
print("Seed: ", args.seed)
print("Best validation ap: ", best_val_ap)
print("Test ap: ", test_ap_at_best_val_ap)