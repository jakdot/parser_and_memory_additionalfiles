"""
Runs parser based on previously executed actions.
"""

import pandas as pd
import simpy
import re
import sys
import numpy as np
import time

import pyactr as actr


from parser_rules import parser
from parser_dm import environment
from parser_dm import SENTENCES
from parser_dm import ACTIVATIONS
import parser_rules
from parser_rules import parser
import utilities as ut
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import BracketParseCorpusReader
from nltk.tree import Tree

# CHANGELOG:
# 28/1/2020 - corrected an issue with recall; when recalling an antecedent, it could be stored that some other antecedent (for example, -TPC) was present; but previously, the recall then always targeted just WP; this is now fixed; however, only one antecedent can be carried at any moment
# 20/8/2019 - added calculations and storing of activations, and sub-parts of activations (fan size, number of matching features, number of agreeing actions)
# 23/4/2019 - added lexical, visual, syntactic, reanalysis as parameters into read function.
# 17/1/2019 - starting the file

WORDS = "words.csv"
LABELS = "labels.csv"

actions, word_freq, label_freq = None, None, None

def visual_effect(word, visual=True):
    if visual:
        return len(word)
    else:
        return 5

def read(parser, sentence=None, pos=None, activations=None, strength_of_association={}, weight=1, threshold=0, decmem={}, lexical=True, visual=True, syntactic=True, reanalysis=True, prints=True, extra_prints=True, condition=None, sent_nr=None):
    """
    Read a sentence.

    :param sentence: what sentence should be read (list).
    :param pos: what pos should be used (list, matching in length with sentence).
    :param activations: dataframe of activations
    :param lexical - should lexical information affect reading time?
    :param visual - should visual information affect reading time?
    :param syntactic - should syntactic information affect reading time?
    :param reanalysis - should reanalysis of parse affect reading time?
    """
    if extra_prints:
        start = time.process_time()
    parser.set_decmem(decmem) 
    parser.decmem.activations = decmem.activations
    tobe_removed = {i for i in range(len(sentence)) if (re.match("[:]+", sentence[i]) or sentence[i] == "'s") and i != len(sentence)-1} #we remove non-words ('s, diacritics)
    if prints:
        print(sentence)
        for x in tobe_removed:
            print(sentence[x])

    if not lexical:
        for x in parser.decmem:
            parser.decmem.activations[x]=100 #this is to nulify the effect of word retrieval to almost 0

    parser.retrievals = {}
    parser.set_retrieval("retrieval")
    parser.visbuffers = {}
    parser.goals = {}
    parser.set_goal("g")
    parser.set_goal(name="imaginal", delay=0)
    parser.set_goal(name="imaginal_reanalysis", delay=0)
    parser.set_goal("word_info")

    stimuli = [{}]
    pos_word = 10 #starting position - just some not so high number higher than 0
    environment.current_focus = (pos_word + 7+7*visual_effect(sentence[0], visual), 180)
    for x in range(41):
        #this removes any move eyes created previously; we assume that no sentence is longer than 20 words
        parser.productionstring(name="move eyes"+ str(x), string="""
        =g>
        isa         reading
        state       dummy
        ==>
        =g>
        isa         reading
        state       dummy""")

    if extra_prints:
        end = time.process_time()
        print("First part", condition, sent_nr, "TIME:", end - start, flush=True)
        start = time.process_time()

    for i, word in enumerate(sentence):
        pos_word += 7+7*visual_effect(word, visual)
        stimuli[0].update({i: {'text': word, 'position': (pos_word, 180), 'vis_delay': visual_effect(word, visual)}})
        
        if i < len(sentence)-3:
            parser.productionstring(name="move eyes" + str(i), string="""
        =g>
        isa             reading
        state            move_eyes
        position        """+str(i)+"""
        ?manual>
        preparation       free
        ==>
        =imaginal>
        isa         action_chunk
        WORD_NEXT0_LEX        """+'"'+str(sentence[i+2])+'"'+"""
        WORD_NEXT0_POS        """+str(pos[i+2])+"""
        WORD_NEXT1_LEX        """+'"'+str(sentence[i+3])+'"'+"""
        WORD_NEXT1_POS        """+str(pos[i+3])+"""
        =g>
        isa             reading
        state   finding_word
        position        """+str(i+1)+"""
        tag             """+str(pos[i+1])+"""
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x    """ + str(pos_word+7+7*visual_effect(sentence[i+1], visual)) + """
        screen_y 180
        ~visual>""")
        elif i < len(sentence)-2:
            parser.productionstring(name="move eyes" + str(i), string="""
        =g>
        isa             reading
        state            move_eyes
        position        """+str(i)+"""
        ?manual>
        preparation       free
        ==>
        =imaginal>
        isa         action_chunk
        WORD_NEXT0_LEX        """+'"'+str(sentence[i+2])+'"'+"""
        WORD_NEXT0_POS        """+str(pos[i+2])+"""
        WORD_NEXT1_LEX        None
        =g>
        isa             reading
        state   finding_word
        position        """+str(i+1)+"""
        tag             """+str(pos[i+1])+"""
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x    """ + str(pos_word+7+7*visual_effect(sentence[i+1], visual)) + """
        screen_y 180
        ~visual>""")
        elif i < len(sentence)-1:
            parser.productionstring(name="move eyes" + str(i), string="""
        =g>
        isa             reading
        state            move_eyes
        position        """+str(i)+"""
        ?manual>
        preparation       free
        ==>
        =imaginal>
        isa         action_chunk
        WORD_NEXT0_LEX        None
        WORD_NEXT1_LEX        None
        =g>
        isa             reading
        state   finding_word
        position        """+str(i+1)+"""
        tag             """+str(pos[i+1])+"""
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x    """ + str(pos_word+7+7*visual_effect(sentence[i+1], visual)) + """
        screen_y 180
        ~visual>""")

    if extra_prints:
        end = time.process_time()
        print("Second part", condition, sent_nr, "TIME:", end - start, flush=True)
        start = time.process_time()

    parser.goals["g"].add(actr.chunkstring(string="""
    isa             reading
    state           reading_word
    position        0
    tag             """+str(pos[0])))

    parser.goals["imaginal"].add(actr.chunkstring(string="""
    isa             action_chunk
    TREE1_LABEL         NOPOS
    TREE1_HEAD          noword
    TREE2_LABEL         xxx
    TREE2_HEAD          xxx
    TREE3_LABEL         xxx
    TREE3_HEAD          xxx
    ANTECEDENT_CARRIED  NO
    WORD_NEXT0_LEX   """+'"'+str(sentence[1])+'"'+"""
    WORD_NEXT0_POS   """+str(pos[1])))
    
    # start a dictionary that will collect all created structures, and a list of built constituents
    built_constituents = [(Tree("xxx", []), (None, "xxx")), (Tree("xxx", []), (None, "xxx")), (Tree("NOPOS", []), (None, "noword"))]
    final_tree = Tree("X", [])

    if prints:
        parser_sim = parser.simulation(realtime=False, gui=False, trace=True, environment_process=environment.environment_process, stimuli=stimuli, triggers='space', times=40)
    else:
        parser_sim = parser.simulation(realtime=False, gui=True, trace=False, environment_process=environment.environment_process, stimuli=stimuli, triggers='space', times=40)

    antecedent_carried = "NO"
    what_antecedent_carried = None

    eye_mvt_times = [] #reaction times, recorded and returned

    reanalyses = [] #reanalysis: 0 - no; 1 - yes

    recall_failures = [] #prob that the parser fails to recall rule

    word_parsed = min(activations['position'])
    last_time = 0
    
    if extra_prints:
        end = time.process_time()
        print(" Third part", condition, sent_nr, "TIME:", end - start, flush=True)
        start = time.process_time()

    while True:
        try:
            parser_sim.step()
            #print(parser_sim.current_event)
        except simpy.core.EmptySchedule:
            eye_mvt_times = [10 for _ in sentence] #if sth goes wrong, it's probably because it got stuck somewhere; in that case report time-out time per word (10 s) or nan
            recall_failures = [1 for _ in sentence] #if sth goes wrong, it's probably because it got stuck somewhere; in that case report failure
            reanalyses = [1 for _ in sentence]
            break
        if parser_sim.show_time() > 60:
            eye_mvt_times = [10 for _ in sentence] #this takes care of looping or excessive time spent - break if you loop (10 s should be definitely enough to move on)
            recall_failures = [1 for _ in sentence] #if sth goes wrong, it's probably because it got stuck somewhere; in that case report failure
            reanalyses = [1 for _ in sentence]
            break
        if re.search("^SHIFT COMPLETE", str(parser_sim.current_event.action)):
            activation = activations[activations['position'].isin([word_parsed])]['activation'].to_numpy()[0]
            extra_rule_time = parser.model_parameters["latency_factor"]*np.exp(-parser.model_parameters["latency_exponent"]*(activation*weight))

            recall_failures.append(1-1/(1+np.exp(-(activation-threshold)/0.4)))

            #reanalysis - adds prob. of regression
            reanalysis = activations[activations['position'].isin([word_parsed])]['reanalysis'].values[0]
            if reanalysis == "yes":
                reanalyses.append(1)
            else:
                reanalyses.append(0)

            # tobe_removed stores positions of parts of words (e.g., 's - we do not calculate RTs on those)
            if len(eye_mvt_times) not in tobe_removed:
                eye_mvt_times.append(parser_sim.show_time() + extra_rule_time - last_time)
            else:
                tobe_removed.remove(len(eye_mvt_times))
            last_time = parser_sim.show_time()
        if re.search(r"^ENCODED VIS OBJECT", str(parser_sim.current_event.action)):
            word_parsed += 1

        if re.search("^RULE FIRED: move attention", str(parser_sim.current_event.action)) and word_parsed >= max( activations['position'] ): #the last word is stopped after move attention - there is nothing to move attention to
            
            activation = activations[activations['position'].isin([word_parsed])]['activation'].to_numpy()[0]
            
            extra_rule_time = parser.model_parameters["latency_factor"]*np.exp(-parser.model_parameters["latency_exponent"]*(activation*weight))
            
            recall_failures.append(1-1/(1+np.exp(-(activation-threshold)/0.4)))

            #reanalysis - adds prob. of regression
            reanalysis = activations[activations['position'].isin([word_parsed])]['reanalysis'].values[0]
            if reanalysis == "yes":
                reanalyses.append(1)
            else:
                reanalyses.append(0)

            if len(eye_mvt_times) not in tobe_removed:
                eye_mvt_times.append(parser_sim.show_time() + extra_rule_time - last_time) #we could eventually add 150 ms to the last word (roughly, pressing the key; this amount of time used in G&G experiment simulation)
            break

        #this below - carrying out an action

        if re.search("^RULE FIRED: recall action", parser_sim.current_event.action) or\
                                re.search("^RULE FIRED: move to last action", parser_sim.current_event.action):
            parser_sim.steps(2) #exactly enough steps to make imaginal full
            
            cg = parser.goals["g"].pop()
            wi = parser.goals["word_info"].copy().pop()
            parser.goals["g"].add(actr.chunkstring(string="""
    isa             reading
    position    """+str(cg.position)+"""
    reanalysis      no
    retrieve_wh     no
    state           finished_recall"""))

            if extra_prints:
                end = time.process_time()
                print("Loop", condition, sent_nr, "TIME:", end - start, flush=True)
                start = time.process_time()
    #final_times = [ eye_mvt_times[0], eye_mvt_times[1], sum(eye_mvt_times[2:]) ] # this would create three measures following Staub - pre-critical word, critical word, spillover
    if prints:
        print("FINAL TIMES")
        print(eye_mvt_times[1:])
    
    if extra_prints:
        end = time.process_time()
        print("End", condition, sent_nr, "TIME:", end - start, flush=True)
        start = time.process_time()

    # return from the first element, because critical=0 is one word before the regions reported in Staub (11)

    if len(eye_mvt_times) == 0:
        eye_mvt_times = [-10 for _ in sentence] #if sth goes wrong, it's probably because it got stuck somewhere; in that case report time-out time per word (10 s) or nan
        recall_failures = [1 for _ in sentence] #if sth goes wrong, it's probably because it got stuck somewhere; in that case report failures
        reanalyses = [1 for _ in sentence]

    return np.array( eye_mvt_times[1:] ), np.array( reanalyses[1:] ), np.array( recall_failures[1:] )
    
if __name__ == "__main__":
    stimuli_csv = ut.load_file(SENTENCES, sep=",") #sentences with frequencies
    words = ut.load_file(WORDS, sep="\t")
    labels = ut.load_file(LABELS, sep="\t")
    activations = ut.load_file(ACTIVATIONS, sep="\t")
    DM = parser.decmem.copy()

    used_activations = activations[activations.critical.isin(["0", "1", "2", "3"])]

    for condition in (set(stimuli_csv.label.to_numpy())):
        condition = "ungram_high"
        print(condition)
        for sent_nr in range(5, 6):
            sent_nr = str(sent_nr)
            print(condition, sent_nr)
            subset_activations = used_activations[used_activations.condition.isin([condition]) & used_activations.item.isin([sent_nr])]
            print(subset_activations)
            subset_stimuli = stimuli_csv[stimuli_csv.label.isin([condition]) & stimuli_csv.item.isin([sent_nr]) & stimuli_csv.position.isin(subset_activations.position.to_numpy())]
            print(subset_stimuli)
            #input()
            final_times = read(parser, sentence=subset_stimuli.word.tolist(), pos=subset_stimuli.function.tolist(), activations=subset_activations, weight=1, decmem=DM, lexical=True, syntactic=True, visual=True, reanalysis=True, prints=True, extra_prints=True, condition=condition, sent_nr=sent_nr)
            print(final_times)
        break

