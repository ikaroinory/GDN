# GDN

A PyTorch implementation using a newer version for the
paper [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series (AAAI'21)](https://arxiv.org/pdf/2106.06947).

## Quick Start

### Requirements

- Python 3.11
- CUDA 11.8

### Installation

Clone the repository:

```bash
git clone https://github.com/ikaroinory/GDN.git
cd GDN
```

Create Conda environment and install dependencies:

```bash
conda create -n GDN python=3.11 -y
conda activate GDN
bash install.sh
```

Run the training script:

```bash
# if you want to preprocess the data:
# python preprocess.py

bash run.sh   # or python main.py
```

## Data

The `data` directory structure should look like this:

```
data
 |- original
 |   |- {dataset_name}
 |   |- ......
 |
 |- processed
 |   |- {dataset_name}
 |   |   |- train.csv
 |   |   |- test.csv
 |   |- ......
```

Your should put data in the `data/processed/{dataset_name}` folder.

## Citation

```
@inproceedings{deng2021graph,
  title     = {Graph neural network-based anomaly detection in multivariate time series},
  author    = {Deng, Ailin and Hooi, Bryan},
  booktitle = {Proceedings of the AAAI conference on artificial intelligence},
  volume    = {35},
  number    = {5},
  pages     = {4027--4035},
  year      = {2021}
}
```
