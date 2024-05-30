# Physics-Informed Deep Learning for Transformer Based Radiotherapy Dose Prediction

This repository contains code that was used to reproduce and extend upon the paper "TrDosePred: A deep learning dose prediction algorithm based on transformers for head and neck cancer radiotherapy". This paper illustrates the application of 3D Vision Transformers in the field of radiation dose treatment planning. In this repository a training pipeline was set up that contains augmentation and preprocessing and allows for plug and play usage of models. For background information and detailed explanation of relevant topics take a look at the [blogpost](blogpost.md).

## Setup

This code was developed using Python 3.11. To setup the project, clone the repository and install the requirements.

```
git clone git@github.com:oxkitsune/DL2.git
pip install -r requirements.txt
```

## Usage

The code can be run using the following command:

```
python -m src.main
```

To use see all available options use the `--help` flag.

## Experiments

This section contains commands that were used to achieve resultst presented in the blogpost.

Experiments for the three-layer UNETR architecture:

```
python -m src.main --batch-size 4 --loss mae --parallel --model unetr --lr 0.0004
python -m src.main --batch-size 4 --loss dvh --parallel --model unetr --lr 0.0004
python -m src.main --batch-size 4 --loss moment --parallel --model unetr --lr 0.0004
python -m src.main --batch-size 4 --loss all --parallel --model unetr --lr 0.0004
```

Experiments for the four-layer UNETR architecture:

```
python -m src.main --batch-size 4 --loss mae --parallel --model bigunetr --lr 0.0004
```

Experiments for the RNN-based architecture:

```
python -m src.main --batch-size 4 --loss mae --parallel --model rnnunetr --lr 0.0004
```
