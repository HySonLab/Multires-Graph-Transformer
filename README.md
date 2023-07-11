# Multiresolution Graph Transformers and Wavelet Positional Encoding for Learning Long-Range and Hierarchical Structures

Paper: https://arxiv.org/pdf/2302.08647.pdf

Contributors:
* Ngo Nhat Khang
* Hy Truong Son (Correspondent / PI)

## Requirements
- [Pytorch](https://pytorch.org/)
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [OGB](https://github.com/snap-stanford/ogb.git)
- [TorchMetrics](https://github.com/Lightning-AI/metrics.git) 
- [Metis](https://anaconda.org/conda-forge/pymetis)
- [PyGSP](https://pygsp.readthedocs.io/en/stable/) \
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

### Protein-Ligand Binding Affinity Datasets
Please download the dataset from [https://zenodo.org/record/4914718](https://zenodo.org/record/4914718), and unzip them to the folder data/lba. \
Then, you can preprocess them by running
```bash
  python3 preprocess.py
```
Finally, to train the model, run:
```bash
  sh scripts/train_lba.sh
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

```bibtex
@inproceedings{townshend2021atomd,
title={{ATOM}3D: Tasks on Molecules in Three Dimensions},
author={Raphael John Lamarre Townshend and Martin V{\"o}gele and Patricia Adriana Suriana and Alexander Derry and Alexander Powers and Yianni Laloudakis and Sidhika Balachandar and Bowen Jing and Brandon M. Anderson and Stephan Eismann and Risi Kondor and Russ Altman and Ron O. Dror},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
year={2021},
url={https://openreview.net/forum?id=FkDZLpK1Ml2}
}
```
