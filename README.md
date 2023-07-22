# Multiresolution Graph Transformers and Wavelet Positional Encoding for Learning Long-Range and Hierarchical Structures

Paper: 
* https://pubs.aip.org/aip/jcp/article/159/3/034109/2903066/Multiresolution-graph-transformers-and-wavelet (Journal of Chemical Physics, Volume 159, Issue 3)
* https://arxiv.org/pdf/2302.08647.pdf (ICML 2023, Workshop of Computational Biology)

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

## Please cite our paper using the following bibtex
```bibtex
@article{10.1063/5.0152833,
    author = {Ngo, Nhat Khang and Hy, Truong Son and Kondor, Risi},
    title = "{Multiresolution graph transformers and wavelet positional encoding for learning long-range and hierarchical structures}",
    journal = {The Journal of Chemical Physics},
    volume = {159},
    number = {3},
    pages = {034109},
    year = {2023},
    month = {07},
    abstract = "{Contemporary graph learning algorithms are not well-suited for large molecules since they do not consider the hierarchical interactions among the atoms, which are essential to determining the molecular properties of macromolecules. In this work, we propose Multiresolution Graph Transformers (MGT), the first graph transformer architecture that can learn to represent large molecules at multiple scales. MGT can learn to produce representations for the atoms and group them into meaningful functional groups or repeating units. We also introduce Wavelet Positional Encoding (WavePE), a new positional encoding method that can guarantee localization in both spectral and spatial domains. Our proposed model achieves competitive results on three macromolecule datasets consisting of polymers, peptides, and protein-ligand complexes, along with one drug-like molecule dataset. Significantly, our model outperforms other state-of-the-art methods and achieves chemical accuracy in estimating molecular properties (e.g., highest occupied molecular orbital, lowest unoccupied molecular orbital, and their gap) calculated by Density Functional Theory in the polymers dataset. Furthermore, the visualizations, including clustering results on macromolecules and low-dimensional spaces of their representations, demonstrate the capability of our methodology in learning to represent long-range and hierarchical structures. Our PyTorch implementation is publicly available at https://github.com/HySonLab/Multires-Graph-Transformer.}",
    issn = {0021-9606},
    doi = {10.1063/5.0152833},
    url = {https://doi.org/10.1063/5.0152833},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0152833/18050074/034109\_1\_5.0152833.pdf},
}
```

```bibtex
@article{ngo2023multiresolution,
  title={Multiresolution Graph Transformers and Wavelet Positional Encoding for Learning Hierarchical Structures},
  author={Ngo, Nhat Khang and Hy, Truong Son and Kondor, Risi},
  journal={arXiv preprint arXiv:2302.08647},
  year={2023}
}
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

```bibtex
@inproceedings{townshend2021atomd,
title={{ATOM}3D: Tasks on Molecules in Three Dimensions},
author={Raphael John Lamarre Townshend and Martin V{\"o}gele and Patricia Adriana Suriana and Alexander Derry and Alexander Powers and Yianni Laloudakis and Sidhika Balachandar and Bowen Jing and Brandon M. Anderson and Stephan Eismann and Risi Kondor and Russ Altman and Ron O. Dror},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
year={2021},
url={https://openreview.net/forum?id=FkDZLpK1Ml2}
}
```
