"""
Estimation of parameters for transition-based parsing.

This is used on Gibson and Grodner (2005) data.

"""

import pyactr as actr
import simpy
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import pymc3 as pm
from pymc3 import Gamma, Normal, HalfNormal, Deterministic, Uniform, find_MAP,\
                  Slice, sample, summary, Metropolis, traceplot
from pymc3.backends.base import merge_traces
from pymc3.backends import Text
from pymc3.backends import load_trace
import theano
import theano.tensor as tt
from mpi4py import MPI
from theano.compile.ops import as_op

print(theano.__version__)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from simpy.core import EmptySchedule

import run_parser as rp
import utilities as ut

WORDS = rp.WORDS
LABELS = rp.LABELS
ACTIVATIONS = rp.ACTIVATIONS

SENTENCES = rp.SENTENCES
parser = rp.parser

subj_extraction = np.array(
        [
         (3, 349.8),
         (4, 354.8),
         (5, 334.3),
         (6, 384),
         (7, 346.5),
         (8, 318.4)], dtype=[('position', np.uint8), ('rt', np.float32)])

obj_extraction = np.array(
        [
         (3, 343),
         (4, 348.1),
         (5, 357.6),
         (6, 422.1),
         (7, 375.8),
         (8, 338.6)], dtype=[('position', np.uint8), ('rt', np.float32)])

def run_self_paced_task():
    """
    Run loop of self paced reading.
    """

    stimuli_csv = ut.load_file(SENTENCES, sep=",") #sentences with frequencies
    words = ut.load_file(WORDS, sep="\t")
    labels = ut.load_file(LABELS, sep="\t")
    activations = ut.load_file(ACTIVATIONS, sep="\t")
    DM = parser.decmem.copy()

    #prepare dictionaries to calculate spreading activation
    word_freq = {k: sum(g["FREQ"].tolist()) for k,g in words.groupby("WORD")} #this method sums up frequencies of words across all POS
    label_freq = labels.set_index('LABEL')['FREQ'].to_dict()

    if rank == 1 or rank == 2:
        condition = "subj_ext"
    elif rank == 3 or rank == 4:
        condition = "obj_ext"

    if rank == 1 or rank == 3:
        sent_nrs = range(1, 9)
    elif rank == 2 or rank == 4:
        sent_nrs = range(9, 17)

    while True:

        received_list = np.empty(4, dtype=np.float)
        comm.Recv([received_list, MPI.FLOAT], source=0, tag=rank)
        if received_list[0] == -1:
            break
        parser.model_parameters["latency_factor"] = received_list[0]
        parser.model_parameters["latency_exponent"] = received_list[1]
        parser.model_parameters["rule_firing"] = received_list[2]
        parser.model_parameters["buffer_spreading_activation"] = {"g": received_list[3]}

        final_times_in_s = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)

        len_sen = 0

        for sent_nr in sent_nrs:
            subset_stimuli = stimuli_csv[stimuli_csv.label.isin([condition]) & stimuli_csv.item.isin([sent_nr])]
            try:
                times_in_s = rp.read(parser, sentence=subset_stimuli.word.tolist(), pos=subset_stimuli.function.tolist(), activations=activations, condition=str(condition),\
                    sent_nr=str(sent_nr), word_freq=word_freq, label_freq=label_freq, weight=received_list[3],\
                    decmem=DM, lexical=True, syntactic=True, visual=False, reanalysis=True, prints=False)
                final_times_in_s += times_in_s
                #print(sent_nr, "FS", times_in_s, flush=True)
            except:
                pass
            else:
                len_sen += 1
        comm.Send([np.array(final_times_in_s/len_sen), MPI.FLOAT], dest=0, tag=1) #len_sen - number of items

@as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector])
def actrmodel_latency(lf, le, rf, weight):
    subj_predicted_ms = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)
    obj_predicted_ms = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)

    sent_list = np.array([lf, le, rf, weight], dtype = np.float)

    print("SENT LIST", sent_list, flush=True)

    #get slaves to work
    for i in range(1, comm.Get_size()):
        comm.Send([sent_list, MPI.FLOAT], dest=i, tag=i)

    subj_idx = 0
    obj_idx = 0

    for i in range(1, N_GROUPS+1):
        #receive list one by one from slaves - each slave - 8 sentences
        received_list = np.empty(6, dtype=np.float)
        comm.Recv([received_list, MPI.FLOAT], i, 1)
        if i == 1 or i == 2:
            subj_predicted_ms += 1000*received_list
        elif i == 3 or i == 4:
            obj_predicted_ms += 1000*received_list

    return [subj_predicted_ms/2, obj_predicted_ms/2]

comm = MPI.COMM_WORLD
rank = int(comm.Get_rank())

N_GROUPS = comm.Get_size() - 1 #Groups used for simulation - one less than used cores

if rank == 0: #master
    NDRAWS = int(sys.argv[1])
    CHAIN = int(sys.argv[2])
    NCHAINS = int(sys.argv[1])

    testval_std, testval_lf, testval_le, testval_rf, testval_weight = 25, 0.1, 0.5, 0.067, 50
    
    # this part collects last estimations if draws were run before
    try:
        past_simulations = ut.load_file("gg_6words_chain"+str(CHAIN)+"/chain-0.csv", sep=",")
    except:
        pass
    else:
        testval_lf = past_simulations['lf'].iloc[-1]
        testval_rf = past_simulations['rf'].iloc[-1]
        testval_le = past_simulations['le'].iloc[-1]
        testval_weight = past_simulations['weight'].iloc[-1]
        testval_std = past_simulations['std'].iloc[-1]

    # the model starts here
    parser_with_bayes = pm.Model()

    with parser_with_bayes:
        # prior for activation
        #decay = Uniform('decay', lower=0, upper=1) #currently, ignored because it leads to problems in sampling
        # priors for latency
        std = Uniform('std', lower=1,upper=50, testval=testval_std)
        lf = Gamma('lf', alpha=2, beta=20, testval=testval_lf) # to get 
        le = Gamma('le', alpha=2,beta=4, testval=testval_le)
        rf = Gamma('rf', alpha=2,beta=30, testval=testval_rf)
        weight = Uniform('weight', lower=1,upper=100, testval=testval_weight)
        # latency likelihood -- this is where pyactr is used
        pyactr_rt = actrmodel_latency(lf, le, rf, weight)
        subj_mu_rt = Deterministic('subj_mu_rt', pyactr_rt[0])
        subj_rt_observed = Normal('subj_rt_observed', mu=subj_mu_rt, sd=std, observed=subj_extraction['rt'])
        obj_mu_rt = Deterministic('obj_mu_rt', pyactr_rt[1])
        obj_rt_observed = Normal('obj_rt_observed', mu=obj_mu_rt, sd=std, observed=obj_extraction['rt'])
        step = Metropolis()
        db = Text('gg_6words_final_chain' + str(CHAIN))
        trace = sample(draws=NDRAWS, trace=db, chains=1, step=step, init='auto', tune=1)

    posterior_checks_subj = pd.DataFrame.from_records(posterior_checks['subj_rt_observed'])
    posterior_checks_obj = pd.DataFrame.from_records(posterior_checks['obj_rt_observed'])
    posterior_checks_subj.to_csv("gg_6words_final_chain" + str(CHAIN) + "/posterior_predictive_checks_subj.csv")
    posterior_checks_obj.to_csv("gg_6words_final_chain" + str(CHAIN) + "/posterior_predictive_checks_obj.csv")

    #stop slaves in their work
    sent_list = np.array([-1, -1, -1], dtype = np.float)
    for i in range(1, comm.Get_size()):
        comm.Send([sent_list, MPI.FLOAT], dest=i, tag=i)

else: #slave
    print(rank)
    run_self_paced_task()
