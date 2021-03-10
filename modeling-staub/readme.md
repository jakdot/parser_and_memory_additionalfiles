# What you can find in this folder

You can run run_parser_act.py to see how the parser parses sentences of Staub (2011):

python3 run_parser_act.py

However, keep in mind that only a small set of chunks for parsing is provided here - based only on a few few hundred sentences of the PTB (the full dataset is not freely available). If you want to get it, please get in touch.

This folder manipulates some basic properties of eye movement which are not allowed in standard ACT-R. You need to install the experimental package pyactr that is provided here to make it run.

You can also run Bayesian model by running parallel_estimation. This requires you to have MPI and run it with 9 parallel jobs, i.e.:

mpirun -n 9 python3 parallel_estimation.py x y

x: parameter specifying number of draws
y: parameter specifying the index of the chain (only one chain is created per 9 jobs)

In the folder "chains used in the paper" you can see the actual draws that were used for the analysis in the paper.
