import torch
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from torch_scatter import scatter
from torch_geometric.utils import degree
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import remove_self_loops

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.L1Loss()

def normalize(train_data, valid_data, test_data):
    train_mean = train_data.data.target[:, :].mean(dim = 0, keepdim = True)
    train_std = train_data.data.target[:, :].std(dim = 0, keepdim = True)
    train_data.data.target = (train_data.data.target - train_mean) / train_std 
    valid_data.data.target = (valid_data.data.target - train_mean) / train_std 
    test_data.data.target = (test_data.data.target - train_mean) / train_std 
    return train_data, valid_data, test_data, train_mean, train_std
    
def train_with_cluster(model, device, loader, optimizer, task_type):
    model.train()
    y_true = []
    y_pred = []
    train_loss = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, link_loss, ent_loss = model(batch)
            pred = pred.squeeze()
            optimizer.zero_grad()
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss = loss + 0.001 * link_loss + 0.001 * ent_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
    return train_loss / len(loader)
    
def eval_with_cluster(model, device, loader, evaluator, task_type = -1):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred, _, _ = model(batch)
                pred = pred.squeeze()
                if task_type == "classification":
                    pred = pred.sigmoid()
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)
    return evaluator(y_pred, y_true)