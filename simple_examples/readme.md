# What you can find in this folder

You can run run_parser_act.py to see how the parser parses simple sentences:

python3 run_parser_act.py

If you want to test your own sentences, check example_sentence.csv. You need to specify:

position (=word position starting from 1), word, function (=POS a la the Penn Treebank),
sentence (=sentence number, starting from 1), critical (=critical region)

In the critical region you can manually specify what rules should be triggered on the word.
See run_parser_act.py, lines 61-64, for details.

Crucially, you have to keep in mind that only a small set of chunks for parsing is provided here - based only on a few few hundred sentences of the PTB (the full dataset is not freely available). If you want to get it, please get in touch with me.

