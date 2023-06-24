# Multiresolution Graph Transformers and Wavelet Positional Encoding for Learning Hierarchical Structures

Paper: https://arxiv.org/pdf/2302.08647.pdf

Contributors:
* Ngo Nhat Khang
* Hy Truong Son (Correspondent / PI)

## Requirements
- [Pytorch](https://pytorch.org/)
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [OGB](https://github.com/snap-stanford/ogb.git)
- [TorchMetrics](https://github.com/Lightning-AI/metrics.git) \
Recommend using Conda for easy installation. 

## Run
### Peptides Datasets
  ```bash
   sh scripts/train_peptides_struct.sh 
  ```
  ```bash
   sh scripts/train_peptides_func.sh 
  ```
### Polymer Datasets
  ```bash
   sh scripts/train_polymer.sh 
  ```
## References
```bibtex
@inproceedings{
  dwivedi2022long,
  title={Long Range Graph Benchmark},
  author={Vijay Prakash Dwivedi and Ladislav Ramp{\'a}{\v{s}}ek and Mikhail Galkin and Ali Parviz and Guy Wolf and Anh Tuan Luu and Dominique Beaini},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=in7XC5RcjEn}
}
```
```bibtex
@article{st2019message,
  title={Message-passing neural networks for high-throughput polymer screening},
  author={St. John, Peter C and Phillips, Caleb and Kemper, Travis W and Wilson, A Nolan and Guan, Yanfei and Crowley, Michael F and Nimlos, Mark R and Larsen, Ross E},
  journal={The Journal of chemical physics},
  volume={150},
  number={23},
  pages={234111},
  year={2019},
  publisher={AIP Publishing LLC}
}
```


device=cuda:2
pe_name=random_walk
batch_size=128
num_layer=2
num_epoch=200
num_head=4
norm=batch
emb_dim=84
num_task=11
dropout=0.25
residual=1
num_cluster=5
attn_dropout=0.5
local_gnn_type=CustomGatedGCN
global_model_type=Transformer
pos_dim=5
trg=0
version=custom
gnn_type=gine # only used for MGT (not CustomMGT)

for seed in 1 2 3 4
do 
    python3 train_polymer.py --device=$device --pe_name=$pe_name --batch_size=$batch_size \
                                   --num_layer=$num_layer --num_epoch=$num_epoch --num_head=$num_head \
                                   --norm=$norm --emb_dim=$emb_dim --num_task=$num_task --dropout=$dropout \
                                   --residual=$residual --num_cluster=$num_cluster --attn_dropout=$attn_dropout \
                                   --local_gnn_type=$local_gnn_type --global_model_type=$global_model_type \
                                   --pos_dim=$pos_dim --version=$version --gnn_type=$gnn_type --trg=$trg
done