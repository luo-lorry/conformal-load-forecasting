# Conformal Load Forecasting on Graphs
This repository contains the official source code for the paper:  
**"Conformal Load Prediction with Transductive Graph Autoencoders"**  
Published in *Machine Learning, Volume 114, Article 54, 2025*.  
DOI: [10.1007/s10994-024-06713-w](https://doi.org/10.1007/s10994-024-06713-w).

## Overview
This repository implements the methods described in the paper for conformal load prediction on graph-structured data using transductive graph autoencoders. The approach leverages conformal prediction techniques to provide uncertainty estimates for edge weight predictions in graphs, with applications in transportation systems and other domains.

## How to Run
To reproduce the results from Table 1 in the paper, run the following command:
```bash
sh Table-1.sh
```

To reproduce the results from Table 2 in the paper, run the following command:
```bash
sh Table-2.sh
```

## Acknowledgement
We acknowledge several GitHub resources that we used in our research. 
- For providing the datasets: https://github.com/bstabler/TransportationNetworks/tree/master;
- For providing the data preprocessing code: https://github.com/000Justin000/ssl_edge;
- For providing the conformal quantile regression code: https://github.com/snap-stanford/conformalized-gnn.

We sincerely appreciate their efforts.

## Citation
If you find this repository or our work helpful, please consider citing our paper:
```
@article{luo2025conformal,
  title={Conformal load prediction with transductive graph autoencoders},
  author={Luo, Rui and Colombo, Nicolo},
  journal={Machine Learning},
  volume={114},
  number={3},
  pages={1--22},
  year={2025},
  publisher={Springer}
}
```
