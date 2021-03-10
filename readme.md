# What you can find in this repository

This repository complements the paper: Parsing as a cue-based retrieval model.

It includes models to replicate the analysis presented in the paper. However, only the sub-symbolic part can be fully replicated (i.e., reading times)! Parsing output will differ from the full parser because this repository does not include full training data set (see Limitations below for details). 

# What you need

The parser works only with Python3 (at least Python 3.3).

It requires various packages:

- pyactr (version 0.2.5 or higher; apart from Staub which requires an experimental pyactr package - see there for details)
- numpy
- nltk
- pandas
- mpi4py
- pymc3

The file in Staub needs a special pyactr experimental package (provided in there).

# simple_examples

This might be the best place to start. It shows how the parser parses some simple sentences. Novel sentences can be provided to the parser.

# train_parser

This folder includes train.py. You can train a parser on your own treebank data using this file. The file will output actions.csv file which stores all the actions that can be used by an ACT-R cue-based parser.

# train_blind_parser

This folder includes train.py. You can train a blind parser on your own treebank data using this file. The file will output blind_actions.csv file which stores all the actions that can be used by an ACT-R cue-based parser that imitates a self-paced reading task (no lookahead). The output from train_parser has a small lookahead (one word).

# modeling-grodnergibson

This file includes files to replicate Bayesian ACT-R models for the self-paced reading experiment of Grodner and Gibson.

# modeling-staub

This file includes files to replicate Bayesian ACT-R models for the eye-tracking experiment of Staub.

# Limitations

The repository only include chunks from the first few hundred sentences of the Penn Treebank. This is because only this part of the Penn Treebank is freely available and publicly available. If you want the full set of chunks, which is used in the paper, you have to train the parser on the full Penn Treebank if you have access to it. Alternatively, please get in touch with me.

