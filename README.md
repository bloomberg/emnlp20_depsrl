# srl-dep

**Author:** Tianze Shi

## About this repo

This repo contains the research code and scripts used in the paper [Semantic Role Labeling as Syntactic Dependency Parsing](https://arxiv.org/abs/2010.11170). This README file aims at giving basic overviews of the code structure and its major components. For more questions, please directly contact the authors.

## Code structure

The entrance point to the package is [here](srl/parser.py). Calls to this package can be chained through fire CLI. The order of calling should usually be `build-vocab`, `create-parser`, `load-embeddings`, `train` and then finally `finish`.
An example inference script is [here](test.py) using `parser.evaluate(data)` after loading in models and embeddings.

For official CoNLL evaluation script, access at [https://www.cs.upc.edu/~srlconll/soft.html](https://www.cs.upc.edu/~srlconll/soft.html). The F1 scores displayed during model training are NOT official F1 scores (though they are usually very close).

### SRL parsing module

The major parsing module is within the python class `SRLDepParser` inside [this file](srl/modules.py). Back-and-forth conversion algorithms tuned on OntoNotes 5.0 data are contained in [this file](srl/conversion.py).

### Pre-trained word embeddings

To speed up loading time, we can process the embedding files to trim down to only the vocabulary seen in our data. Script for trimming is [here](srl/filter_embeddings.py).

### Data preparation

Data preparation scripts lie under `data_prep` folder.

Prerequisite: [Stanford CoreNLP with English and Chinese models v3.9.2](https://stanfordnlp.github.io/CoreNLP/history.html)

1. Follow http://cemantix.org/data/ontonotes.html to prepare data, using v12 data release
2. For Chinese data (http://conll.cemantix.org/2012/data.html), copy folders under the correct splits
3. Use train dev and conll-2012-test splits for English, train dev, test for Chinese
4. Run `aggregate.sh`
5. Run `space_to_tab.sh`
6. Run `constituency_tree.sh`
7. Run `english_dep_tree.sh` and `english_fuse.py` for English data preparation
8. Run `chinese_dep_tree.sh` and `chinese_fuse.py` for Chinese data preparation
