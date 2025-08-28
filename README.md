# Neural Logic Networks for Interpretable Tabular Classification

## Description

This repository implements the code for the paper [Neural Logic Networks for Interpretable Classification](https://arxiv.org/abs/2508.08172), not yet published.

## Requirements

This project requires Python3 and [PyTorch](https://pytorch.org/get-started/locally/) (ideally with CUDA support for parallelization in the learning process).

## Getting started

### Installation

First, clone the repository.

```
$ git clone https://https://github.com/pedrosbmartins/optimal-decision-graphs.git
```

Then, create and activate a python environment (for instance, using [virtualenv](https://virtualenv.pypa.io/en/latest/)).

```
$ virtualenv -p python3 new_env
$ source new_env/bin/activate
```

Finally, install required packages.

```
$ pip install -r requirements.txt
$ pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Basic usage

> Note: make sure your dataset is formatted as a `.csv` file in the `datasets` directory.

Follow the example codes depending on your situation:
- `NLN_example_train.py` if you only have a training dataset file (including validation) on which to learn a NLN;
- `NLN_example_train-test.py` if you have both a training dataset file (including validation) to learn a NLN and a test dataset file to test it;
- `NLN_example_5-fold-cross-validation.py` if you wish to train, test and cross-validate on the same dataset file;
- `NLN_example_merge.py` if you wish to merge multiple learned models (for instance, after 5-fold cross-validation).

The first time any `.csv` dataset file is used, prompts in the terminal will guide you to determine the types of your features (binary, categorical or continuous/ordinal) as well as which column(s) should be the target(s) for classification (binary or categorical). If reformatting of the `.csv` file iteself is required, a copy will be created.

This current implementation of NLNs allows 3 use-cases:
1. Binary Classification - a single binary target;
2. Multi-Class Classification - a single categorical target;
3. Multi-Label Classifiaction - multiple binary targets.