seed=5
num_epoch=60
batch_size=128
num_cluster=20
emb_dim=84
num_layer=2
device=cuda:6
dropout=0.25
attn_dropout=0.5
batch_norm=0
layer_norm=1
local_gnn_type=CustomGatedGCN
global_model_type=None
num_task=1
data_format=ogb
pe_name=wave
diff_step=5
equiv_pe=0

python3 train_lba.py --seed=$seed --num_epoch=$num_epoch --batch_size=$batch_size \
                     --num_cluster=$num_cluster --dropout=$dropout --attn_dropout=$attn_dropout \
                     --emb_dim=$emb_dim --batch_norm=$batch_norm --layer_norm=$layer_norm \
                     --device=$device --local_gnn_type=$local_gnn_type \
                     --global_model_type=$global_model_type --num_task=$num_task \
                     --data_format=$data_format --pe_name=$pe_name --diff_step=$diff_step