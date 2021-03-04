"""
Parse: dm.
"""

import pyactr as actr
import simpy
import re
import sys

import numpy as np
import pandas as pd

SENTENCES = "sentences_exp3.csv"
ACTIVATIONS = "activation_sentences_exp3.csv"

SEC_IN_YEAR = 365*24*3600
SEC_IN_TIME = 15*SEC_IN_YEAR

def load_file(lfile, index_col=None, sep=","):
    """
    Loads file as a list
    """
    csvfile = pd.read_csv(lfile, index_col=index_col, header=0, sep=sep)
    return csvfile

def get_freq_array(word_freq, real_freq=True):
    """
    Finds the frequency, based on the dataframe df.
    """
    time_interval = SEC_IN_TIME / word_freq
    if real_freq:
        return np.arange(start=-time_interval, stop=-(time_interval*word_freq)-1, step=-time_interval)
    else:
        return np.array([0])

def calculate_activation(parser, freq_array):
    """
    Calculate activation using parser and freq_array.
    """
    return np.log(np.sum(freq_array ** (-parser.model_parameters["decay"])))

stimuli_csv = load_file(ACTIVATIONS, sep="\t") #sentences with frequencies
sentences_csv = load_file(SENTENCES, sep=",") #sentences with frequencies

environment = actr.Environment(focus_position=(320, 180))
parser = actr.ACTRModel(environment, subsymbolic=True, retrieval_threshold=-20,
                        decay = 0.5, latency_factor=0.01, latency_exponent=0.01,
                        eye_mvt_angle_parameter=1, emma_noise=False,
                        eye_mvt_scaling_parameter=0.01,
                        emma_preparation_time=0.09,
                        rule_firing=0.02, motor_prepared=True, automatic_visual_search=False)
temp_dm = {}

temp_activations = {}

stimuli_csv = stimuli_csv[stimuli_csv.critical.isin(["0", "1", "2", "3"])]

words = stimuli_csv.groupby('word', sort=False)

actr.chunktype("word", "form cat")

actr.chunktype("parsing_goal", "task")

actr.chunktype("reading", "state position word reanalysis retrieve_wh what_retrieve tag")


actr.chunktype("action_chunk", "ACTION ACTION_RESULT_LABEL ACTION_PREV WORD_NEXT0_LEX WORD_NEXT0_POS TREE0_LABEL TREE1_LABEL TREE2_LABEL TREE3_LABEL TREE0_HEAD TREE0_HEADPOS TREE0_LEFTCHILD TREE0_RIGHTCHILD TREE1_HEAD TREE1_HEADPOS TREE1_LEFTCHILD TREE1_RIGHTCHILD TREE2_HEAD TREE2_HEADPOS TREE3_HEAD ANTECEDENT_CARRIED")

for name, group in words:
        word = group.iloc[0]['word']
        function = sentences_csv[sentences_csv.word.isin([word])].function.to_numpy()[0]
        freq = sentences_csv[sentences_csv.word.isin([word])].freq.to_numpy()[0]
        temp_dm[actr.chunkstring(string="""
        isa  word
        form """+'"'+str(word)+'"'+"""
        cat  """+str(function)+"""
        """)] = np.array([])
        temp_activations[actr.chunkstring(string="""
        isa  word
        form """+'"'+str(word)+'"'+"""
        cat  """+str(function)+"""
        """)] = calculate_activation(parser, 0-get_freq_array(freq))

parser.decmems = {}
parser.set_decmem(temp_dm)

for elem in parser.decmem:
    parser.decmem.activations[elem] = temp_activations[elem]
