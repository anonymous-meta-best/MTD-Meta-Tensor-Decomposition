# MTD: Meta Tensor Decomposition

This is a python implementation of the paper 
"Meta-Learning for Fast and Accurate Domain Adaptation for Irregular Tensors".

## Prerequisites

- Python 3.5+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [sciket-learn](https://scikit-learn.org/)

## Usage

You can run a demo script `run.sh` that reproduces the experimental results in the paper by the following command.
You can change the hyperparameters by modifying `run.sh`.
```
bash run.sh
```

## Datasets

Preprocessed data are included in the `data` directory.
You can use your own data if it is a 3-way irregular tensors in multiple domains.

| Name           | # Domain |   Max $I_k$ |  $(J, K)$ | # Non-z. | Summary | Download                                                        |
|----------------|---------:|------------:|----------:|---------:|--------:|:----------------------------------------------------------------|
| NATOPS-H       |       20 |       2,009 |  (77, 24) |  42,720K |     HAR | https://github.com/yalesong/natops                              |
| NATOPS         |        6 |          51 |  (24, 60) |     440K |     HAR | https://timeseriesclassification.com                            |
| Cricket        |       12 |       1,197 |   (6, 10) |   1,292K |     HAR | https://timeseriesclassification.com                            |
| FingerMovement |        2 |          50 | (28, 208) |     582K |     EEG | https://timeseriesclassification.com                            |
| PEMS-SF        |        7 |         144 | (963, 57) |  55,330K |  subway | https://timeseriesclassification.com                            |
| Nasdaq         |        3 |      12,709 |   (6, 11) |   2,742K |   stock | https://kaggle.com/datasets/paultimothymooney/stock-market-data |
| SP500          |       11 |      13,321 |   (6, 13) |   7,318K |   stock | https://kaggle.com/datasets/paultimothymooney/stock-market-data |
| Korean stock   |       11 |       3,089 |   (6, 10) |   2,038K |   stock | https://github.com/jungijang/KoreaStockData                     |
