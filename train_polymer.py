import torch_geometric
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

#from gnn import GNN
from torch_geometric.transforms import AddRandomWalkPE, AddLaplacianEigenvectorPE
import argparse
from utils import *
from tqdm import tqdm
from metrics import *
from model import *
from polymer_dataset import PolymerDataset
from wave_pe import WaveletPE, Learnable_Equiv_WaveletPE



class MyTransform(object):
    def __init__(self, mean, std, trg, pos_dim):
        self.mean = mean 
        self.std = std 
        self.trg = trg
        self.pos_dim = pos_dim

    def __call__(self, data):
        # Specify target.
        data.target = data.target[:, self.trg]
        data.target = (data.target - self.mean) / self.std
        if data.pos is not None:
            data.pos = data.pos[:, : self.pos_dim]
        return data

root = "dataset"
name = "full"
targets = ["gap", "humo", "lumo"]

parser = argparse.ArgumentParser(description="GNN on Polymer")
parser.add_argument("--out_dir", type = str, default = "./results")
parser.add_argument("--seed", type = int, default=123456)
parser.add_argument("--pe_name", type = str, default = "lap")
parser.add_argument("--device", type=str, default="cuda:0", help = "cuda device")
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
parser.add_argument("--trg", type = int, default = 0)
parser.add_argument("--pos_dim", type = int, default= 8)
parser.add_argument("--version", type = str, default = "custom")

args = parser.parse_args()

torch_geometric.seed_everything(args.seed)

#wandb.init(project = "Polymer Project")
#wandb.run.name = f"{args.model_name}_{args.seed}_{targets[args.trg]}"

# get train_mean and std_mean
data = PolymerDataset(name, root, "train", transform= None)

print("Target: ", targets[args.trg])

mean = data.data.target[:, args.trg].mean(dim = 0, keepdim = True)
std = data.data.target[:, args.trg].std(dim = 0, keepdim = True)

if args.pe_name == "laplacian":
    lap_transforms = AddLaplacianEigenvectorPE(args.pos_dim, "pos", True)
    transforms = T.Compose([MyTransform(mean, std, args.trg, args.pos_dim), lap_transforms])
    train_data = PolymerDataset(name, root, "train", transform=transforms)
    valid_data = PolymerDataset(name, root, "valid", transform=transforms)
    test_data = PolymerDataset(name, root, "test", transform=transforms)
elif args.pe_name == "random_walk":
    rw_transforms = AddRandomWalkPE(args.pos_dim, "pos")
    transforms = T.Compose([MyTransform(mean, std, args.trg, args.pos_dim), rw_transforms])
    train_data = PolymerDataset(name, root, "train", transform=transforms)
    valid_data = PolymerDataset(name, root, "valid", transform=transforms)
    test_data = PolymerDataset(name, root, "test", transform=transforms)
elif args.pe_name == "wave":
    pre_transforms = WaveletPE(args.pos_dim)
    transforms = T.Compose([MyTransform(mean, std, args.trg, args.pos_dim)])
    train_data = PolymerDataset(name, root, "train", transform=transforms, pre_transform=pre_transforms)
    valid_data = PolymerDataset(name, root, "valid", transform= transforms, pre_transform= pre_transforms)
    test_data = PolymerDataset(name, root, "test", transform=transforms, pre_transform=pre_transforms)
elif args.pe_name == "learnable_equiv_wave":
    pre_transforms = Learnable_Equiv_WaveletPE(args.pos_dim)
    transforms = T.Compose([MyTransform(mean, std, args.trg, args.pos_dim)])
    train_data = PolymerDataset(name, root, "train", transform=transforms, pre_transform=pre_transforms)
    valid_data = PolymerDataset(name, root, "valid", transform= transforms, pre_transform= pre_transforms)
    test_data = PolymerDataset(name, root, "test", transform=transforms, pre_transform=pre_transforms)
else:
    raise NotImplemented

print("Num train: ", len(train_data))
print("Num valid: ", len(valid_data))
print("Num test: ", len(test_data))


train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, num_workers = 6, pin_memory = False)
valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, num_workers = 6, pin_memory = False)
test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, num_workers = 6, pin_memory = False)

device = args.device
num_layer = args.num_layer
num_head = args.num_head
num_task = args.num_task
pre_cluster = bool(args.pre_cluster)
use_multi = bool(args.use_multi)
args.equiv_pe = True

if args.version == "custom":
    model = CustomMGT(args, args.pe_name).to(args.device)
else:
    model = MGT(args.num_layer, args.emb_dim, args.pos_dim, args.num_task, args.num_head, args.dropout, 
                args.attn_dropout, args.norm, args.num_cluster, args.gnn_type, args.pe_name, args.device)

print("Number of parameters: ", model.num_parameters)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay=0)
scheduler = torch.optim.lr_scheduler. ReduceLROnPlateau(optimizer, 'min', min_lr = 1e-5)

evaluator = MetricWrapper("mae")
best_val_mae = 1e9
test_mae_at_best_val_mae = -1

for epoch in range(1, args.num_epoch + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    train_mae = train_with_cluster(model, device, train_loader, optimizer, ["regression"])

    print('Evaluating...')
    valid_mae = eval_with_cluster(model, device, valid_loader, evaluator, "regression")
    test_mae = eval_with_cluster(model, device, test_loader, evaluator, "regression")
    
    scheduler.step(valid_mae)

    if best_val_mae > valid_mae:
        best_val_mae = valid_mae
        test_mae_at_best_val_mae = test_mae

    print('| Train mae: {:5.5f} | Validation mae: {:5.5f} | Test mae: {:5.5f}'.format(train_mae, valid_mae, test_mae))

print("*" * 20)
print(args)
print()
print("Number of parameters: ", model.count_params())
print("Best validation mae: ", best_val_mae)
print("Test mae: ", test_mae_at_best_val_mae)