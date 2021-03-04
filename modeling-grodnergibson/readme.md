# What you can find in this folder

You can run run_parser_act.py to see how the parser parses sentences of Grodner and Gibson (2005):

python3 run_parser_act.py

However, keep in mind that in this illustrative case, we only use chunks based on the first few hundred sentences of the PTB (the full dataset is not freely available).

You can also run Bayesian model by running parallel_estimation. This requires you to have MPI and run it with 5 parallel jobs, i.e.:

mpirun -n 5 python3 parallel_estimation.py x y

x: parameter specifying number of draws
y: parameter specifying the index of the chain (only one chain is created per 5 jobs)

In the folder "chains used in the paper" you can see the actual draws that were used for the analysis in the paper.
