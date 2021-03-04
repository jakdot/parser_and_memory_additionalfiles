"""
Estimation of parameters for transition-based parser.

This is used on Staub (2011) data.

"""

import pyactr as actr
import simpy
import collections
from scipy.stats import binom
import re
import sys
import warnings
import time
import pympler
from pympler import muppy
warnings.filterwarnings("ignore")

import pymc3 as pm
from pymc3 import Gamma, Normal, HalfNormal, Deterministic, Uniform, find_MAP,\
                  Slice, sample, summary, Metropolis, traceplot
from pymc3.backends.base import merge_traces
from pymc3.backends import Text
from pymc3.backends.sqlite import load
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

COUNTING = 0

SENTENCES = rp.SENTENCES
parser = rp.parser

observed = np.array(
        [
        ("1-good-high", 237, .06),
        ("2-good-high", 266, .13),
        ("3-good-high", 810, .40),
         ("1-good-low", 239, .07),
         ("2-good-low", 306, .17),
         ("3-good-low", 765, .46),
         ("1-bad-high", 249, .16),
         ("2-bad-high", 322, .29),
         ("3-bad-high", 675, .59),
         ("1-bad-low", 252, .12),
         ("2-bad-low", 340, .34),
         ("3-bad-low", 730, .52)], dtype=[('position-condition', np.str), ('rt', np.float32), ('reg', np.float32)])

def run_reading_task():
    """
    Run loop of self paced reading.
    """

    def calculate_times(times_in_s, reanalyses, recall_failures, prob_regression):
        """
        Helper function to calculate first pass times on last region (several words, so regression could be triggered; when regression is triggered, first-pass RT is stopped).
        """
        final_time = times_in_s[0]

        number_of_regressions = 0

        recall = 1-recall_failures[0]

        for i in range(1, len(times_in_s)):
            if reanalyses[i-1] == 1:
                number_of_regressions += 1
            recall *= 1-recall_failures[i]
            final_time += times_in_s[i]*(recall*(1-prob_regression)**number_of_regressions) #only non-regressed RTs are added
        return final_time

    stimuli_csv = ut.load_file(SENTENCES, sep=",") #sentences with frequencies
    activations = ut.load_file(ACTIVATIONS, sep="\t")
    DM = parser.decmem.copy()

    if rank == 1 or rank == 2:
        condition = "gram_high"
    elif rank == 3 or rank == 4:
        condition = "gram_low"
    elif rank == 5 or rank == 6:
        condition = "ungram_high"
    elif rank == 7 or rank == 8:
        condition = "ungram_low"

    if rank%2 == 1:
        sent_nrs = range(1, 12)
    else:
        sent_nrs = range(12, 23)

    used_activations = activations[activations.critical.isin(["0", "1", "2", "3"])]

    del activations

    COUNTING = 0

    while True:
        received_list = np.empty(6, dtype=np.float)
        comm.Recv([received_list, MPI.FLOAT], source=0, tag=rank)
        if received_list[0] == -1:
            break
        parser.model_parameters["latency_factor"] = received_list[0]
        parser.model_parameters["latency_exponent"] = received_list[1]
        parser.model_parameters["rule_firing"] = received_list[2]
        parser.model_parameters["emma_preparation_time"] = received_list[3]
        prob_regression = received_list[4]
        threshold = received_list[5]

        del received_list
        #print(rank, condition, sent_nrs)

        final_times_in_s = np.array([0, 0, 0], dtype=np.float)
        regressions = np.array([0, 0, 0], dtype=np.float)

        len_sen = 0
        extra_prints = False

        for sent_nr in sent_nrs:
            subset_activations = used_activations[used_activations.condition.isin([condition]) & used_activations.item.isin([sent_nr])]
            subset_stimuli = stimuli_csv[stimuli_csv.label.isin([condition]) & stimuli_csv.item.isin([sent_nr]) & stimuli_csv.position.isin(subset_activations.position.to_numpy())]
            try:
                if COUNTING % 200 == 0:
                    start = time.process_time()
                times_in_s, reanalyses, recall_failures = rp.read(parser, sentence=subset_stimuli.word.tolist(), pos=subset_stimuli.function.tolist(), activations=subset_activations, weight=10, threshold=threshold,\
                    decmem=DM.copy(), lexical=True, syntactic=True, visual=True, reanalysis=True, prints=False, extra_prints=extra_prints, condition=condition, sent_nr=sent_nr)
                if COUNTING % 200 == 0:
                    end = time.process_time()
                    start = time.process_time()

                regressions += np.array([recall_failures[0] + prob_regression*reanalyses[0], recall_failures[1] + prob_regression*reanalyses[1], min(1, 1-binom.pmf(0, n=len(times_in_s[2:]), p=np.mean(recall_failures[2:])) + 1-binom.pmf(0, n=sum(reanalyses[2:]), p=prob_regression))], dtype=np.float) #calculate prob. of regressions for 3 regions in Staub: 0 (pre-critical), 1 (critical), 2 (all later words; post-critical)
                final_times_in_s += np.array([times_in_s[0], times_in_s[1], calculate_times(times_in_s[2:], reanalyses[2:], recall_failures[2:], prob_regression) ], dtype=np.float)
            except:
                pass
            else:
                len_sen += 1
        
        COUNTING += 1

        to_be_sent = np.append( final_times_in_s/len_sen, regressions/len_sen )
        comm.Send([to_be_sent, MPI.FLOAT], dest=0, tag=1) #len_sen - number of items

@as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar], otypes=[tt.dvector, tt.dvector])
def actrmodel_latency(lf, le, emma_prep_time, prob_regression, threshold):
    global COUNTING
    gram_high_predicted_ms = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)
    gram_low_predicted_ms = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)
    ungram_high_predicted_ms = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)
    ungram_low_predicted_ms = np.array([0, 0, 0, 0, 0, 0], dtype=np.float)

    rf = 0.03 #we set rf at this value - close to the value from Grodner & Gibson

    sent_list = np.array([lf, le, rf, emma_prep_time, prob_regression, threshold], dtype = np.float)

    if COUNTING % 200 == 0:
        start = time.process_time()
        print("SENT LIST", sent_list)

    #get slaves to work
    for i in range(1, comm.Get_size()):
        comm.Send([sent_list, MPI.FLOAT], dest=i, tag=i)

    #print("STARTING", sent_list)

    change_to_s = np.array([1000, 1000, 1000, 1, 1, 1])

    for i in range(1, N_GROUPS+1):
        #receive list one by one from slaves - each slave - 8 sentences
        received_list = np.empty(6, dtype=np.float)
        comm.Recv([received_list, MPI.FLOAT], i, 1)
        if i == 1 or i == 2:
            gram_high_predicted_ms += change_to_s*received_list
        elif i == 3 or i == 4:
            gram_low_predicted_ms += change_to_s*received_list
        elif i == 5 or i == 6:
            ungram_high_predicted_ms += change_to_s*received_list
        elif i == 7 or i == 8:
            ungram_low_predicted_ms += change_to_s*received_list

    if COUNTING % 200 == 0:
        print(gram_high_predicted_ms/2, gram_low_predicted_ms/2, ungram_high_predicted_ms/2, ungram_low_predicted_ms/2, flush=True)
        end = time.process_time()
        print("TIME:", end - start, flush=True)

    COUNTING += 1

    #the first 3 - RTs; the last 3 - prob. regressions

    final_rts = np.concatenate([gram_high_predicted_ms[:3]/2, gram_low_predicted_ms[:3]/2, ungram_high_predicted_ms[:3]/2, ungram_low_predicted_ms[:3]/2])

    final_reg = np.concatenate([gram_high_predicted_ms[3:]/2, gram_low_predicted_ms[3:]/2, ungram_high_predicted_ms[3:]/2, ungram_low_predicted_ms[3:]/2])

    return [final_rts, final_reg]


comm = MPI.COMM_WORLD
rank = int(comm.Get_rank())

N_GROUPS = comm.Get_size() - 1 #Groups used for simulation - one less than used cores

if rank == 0: #master
    NDRAWS = int(sys.argv[1])
    CHAIN = int(sys.argv[2])

    testval_std, testval_lf, testval_le = 25, 0.1, 0.5

    testval_threshold, testval_emma_prep_time, testval_prob_regression = 0, 0.133, 0.25

    # collect previous values from the existing chains

    try:
        past_simulations = ut.load_file("staub_exp3_chain"+str(CHAIN)+"/chain-0.csv", sep=",")
    except:
        pass
    else:
        testval_lf = past_simulations['lf'].iloc[-1]
        testval_le = past_simulations['le'].iloc[-1]
        testval_threshold = past_simulations['threshold'].iloc[-1]
        testval_emma_prep_time = past_simulations['emma_prep_time'].iloc[-1]
        testval_prob_regression = past_simulations['prob_regression'].iloc[-1]
        testval_std = past_simulations['std'].iloc[-1]

    # here the model starts

    parser_with_bayes = pm.Model()

    with parser_with_bayes:
        # prior for activation
        #decay = Uniform('decay', lower=0, upper=1) #currently, ignored because it leads to problems in sampling
        # priors for latency
        std = Uniform('std', lower=1,upper=60, testval=testval_std)
        lf = Gamma('lf', alpha=2, beta=20, testval=testval_lf)
        le = Gamma('le', alpha=2,beta=4, testval=testval_le)
        threshold = Normal('threshold', mu=0, sd=10, testval=testval_threshold)
        emma_prep_time = Gamma('emma_prep_time', alpha=4, beta=30, testval=testval_emma_prep_time)
        prob_regression = Uniform('prob_regression', lower=0.01, upper=0.5, testval=testval_prob_regression)
        # latency likelihood -- this is where pyactr is used
        pyactr_rt = actrmodel_latency(lf, le, emma_prep_time, prob_regression, threshold)
        #RTs
        mu_rt = Deterministic('mu_rt', pyactr_rt[0])
        rt_observed = Normal('rt_observed', mu=mu_rt, sd=std, observed=observed['rt'])
        #regressions
        mu_reg = Deterministic('mu_reg', pyactr_rt[1])
        reg_observed = Normal('reg_observed', mu=mu_reg, sd=0.05, observed=observed['reg'])
        # we start the sampling
        step = Metropolis()
        db = Text('staub_exp3_chain' + str(CHAIN))
        trace = sample(draws=NDRAWS, trace=db, chains=1, step=step, init='auto', tune=1)
        traceplot(trace)
        plt.savefig("staub_posteriors_chain" + str(CHAIN) + ".pdf")
        plt.savefig("staub_posteriors_chain" + str(CHAIN) + ".png")

    #stop slaves in their work
    sent_list = np.array([-1, -1, -1], dtype = np.float)
    for i in range(1, comm.Get_size()):
        comm.Send([sent_list, MPI.FLOAT], dest=i, tag=i)

else: #slave
    print(rank)
    run_reading_task()
