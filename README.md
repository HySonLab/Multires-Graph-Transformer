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
@article{ngo2023multiresolution,
  title={Multiresolution Graph Transformers and Wavelet Positional Encoding for Learning Hierarchical Structures},
  author={Ngo, Nhat Khang and Hy, Truong Son and Kondor, Risi},
  journal={arXiv preprint arXiv:2302.08647},
  year={2023}
}

```

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
                                   --norm=$norm --emb_dim=$emb_dim --num_task=$num_task --dropout=$dropout \
                                   --residual=$residual --num_cluster=$num_cluster --attn_dropout=$attn_dropout \
                                   --local_gnn_type=$local_gnn_type --global_model_type=$global_model_type \
                                   --pos_dim=$pos_dim --version=$version --gnn_type=$gnn_type --trg=$trg
done
