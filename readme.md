# What you can find in this repository

This repository complements the paper: Parsing as a cue-based retrieval model.

It includes models to replicate the analysis presented in the paper.

# train_parser

This folder includes train.py. You can train a parser on your own treebank data using this file. The file will output actions.csv file which store all the actions that can be used by an ACT-R cue-based parser.

# train_blind_parser

This folder includes train.py. You can train a blind parser on your own treebank data using this file. The file will output blind_actions.csv file which store all the actions that can be used by an ACT-R cue-based parser that imitates a self-paced reading task (no lookahead).

# modeling-grodnergibson

This file includes files to replicate Bayesian ACT-R models for the self-paced reading experiment of Grodner and Gibson.

# modeling-staub

This file includes files to replicate Bayesian ACT-R models for the eye-tracking experiment of Staub.

# Limitations

The repository only include chunks from the first few hundred sentences of the Penn Treebank. This is because only this part of the Penn Treebank is freely available in nltk. If you want the full set of chunks, which is used in the paper, you have to train the parser on the full Penn Treebank if you have access to it. Alternatively, please get in touch with me.

