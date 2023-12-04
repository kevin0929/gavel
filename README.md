# Gavel : The inspector of Cow

## Where to get it

You can download source code by Git

```
git clone https://github.com/kevin0929/Gavel.git
```

Strongly recommend you to use `pyenv` or `conda`

```
conda create -n gavel python=3.9
conda activate gavel
```

## Installation

Install dependencies by `requirements.txt` in `python >= 3.9` environment, including `Pytorch >= 2.0`

```
cd gavel
pip install -r requirements.txt
```

## Utils

- `process.py` : For data preprocess, use cwt to transform sensor data to image-based data (npz format).

- `model.py` : The architecture of GavelModel.

- `train.py` : For training, detail : 
    - batch size: 32
    - learning rate: 0.001
    - epochs: 150
    - loss function: cross entropy
    - optimizer: Adam
