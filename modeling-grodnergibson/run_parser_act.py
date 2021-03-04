"""
Runs parser based on previously executed actions.
"""

import pandas as pd
import simpy
import re
import sys
import numpy as np

import pyactr as actr

from parser_rules import parser
from parser_dm import environment
from parser_dm import SENTENCES
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
ACTIONS = "actions.csv"
BLIND_ACTIONS = "blind_actions.csv"

actions, word_freq, label_freq = None, None, None

def visual_effect(word, visual=True):
    if visual:
        return len(word)
    else:
        return 5

def read(parser, sentence=None, pos=None, actions=actions, blind_actions=actions, word_freq=word_freq, label_freq=label_freq, strength_of_association={}, decmem={}, lexical=True, visual=True, syntactic=True, reanalysis=True, prints=True):
    """
    Read a sentence.

    :param sentence: what sentence should be read (list).
    :param pos: what pos should be used (list, matching in length with sentence).
    :param actions: dataframe of actions
    :param lexical - should lexical information affect reading time?
    :param visual - should visual information affect reading time?
    :param syntactic - should syntactic information affect reading time?
    :param reanalysis - should reanalysis of parse affect reading time?
    """

    strength_of_association_reanalysis = {"WORD_NEXT0_LEX": 25, "WORD_NEXT0_POS": 25} #increased because spr - only 1 word; if we had WORD_NEXT1 also, the fit is better; so this is useful to increase the role of upcoming info (for reanalysis)

    parser.set_decmem(decmem) # TODO: dont remove??
    tobe_removed = {i for i in range(len(sentence)) if (re.match("[:]+", sentence[i]) or sentence[i] == "'s") and i != len(sentence)-1}
    print(sentence)
    for x in tobe_removed:
        print(sentence[x])

    if not lexical:
        for x in parser.decmem:
            parser.decmem.activations[x]=100 #this is to nullify the effect of word retrieval to almost 0

    parser.retrievals = {}
    parser.set_retrieval("retrieval")
    parser.visbuffers = {}
    parser.goals = {}
    parser.set_goal("g")
    parser.set_goal(name="imaginal", delay=0)
    parser.set_goal(name="imaginal_reanalysis", delay=0)
    parser.set_goal("word_info")

    stimuli = [{} for i in range(len(sentence))]
    pos_word = 10
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
    for i, word in enumerate(sentence):
        pos_word += 7+7*visual_effect(word, visual)
        for j in range(len(stimuli)):
            if j == i:
                stimuli[j].update({i: {'text': word, 'position': (pos_word, 180), 'vis_delay': visual_effect(word, visual)}})
            else:
                stimuli[j].update({i: {'text': "___", 'position': (pos_word, 180), 'vis_delay': 3}})
        
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
        =g>
        isa             reading
        state   reading_word
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
        =g>
        isa             reading
        state   reading_word
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
        =g>
        isa             reading
        state   reading_word
        position        """+str(i+1)+"""
        tag             """+str(pos[i+1])+"""
        ?visual_location>
        attended False
        +visual_location>
        isa _visuallocation
        screen_x    """ + str(pos_word+7+7*visual_effect(sentence[i+1], visual)) + """
        screen_y 180
        ~visual>""")


    if prints:
        print(sentence)

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
    constituents = {}
    built_constituents = [(Tree("xxx", []), (None, "xxx")), (Tree("xxx", []), (None, "xxx")), (Tree("NOPOS", []), (None, "noword"))]
    final_tree = Tree("X", [])

    if prints:
        parser_sim = parser.simulation(realtime=False, gui=False, trace=True, environment_process=environment.environment_process, stimuli=stimuli, triggers='space', times=40)
    else:
        parser_sim = parser.simulation(realtime=False, gui=True, trace=False, environment_process=environment.environment_process, stimuli=stimuli, triggers='space', times=40)

    antecedent_carried = "NO"
    what_antecedent_carried = None

    spr_times = [] #reaction times, recorded and returned
    reanalysis_list, words_list, activations_list, agreeing_actions_list, matching_fs_list, total_fan_list, actions_list = [], [], [], [], [], [], [] #collects total activation of rules, agreeing_actions... (used to find out what plays a role in syntactic parsing for RTs)
    wh_gaps_list = []

    word_parsed = 0
    last_time = 0

    activations, agreeing_actions, matching_fs, total_fan = [], [], [], [] #collects total activation of rules, agreeing_actions... (used to find out what plays a role in syntactic parsing for RTs)

    retrieve_wh_reanalysis = None

    while True:
        try:
            parser_sim.step()
            #print(parser_sim.current_event)
        except simpy.core.EmptySchedule:
            spr_times = [10 for _ in sentence] #if sth goes wrong, it's probably because it got stuck somewhere; in that case report time-out time per word (40 s) or nan
            break
        if parser_sim.show_time() > 60:
            spr_times = [10 for _ in sentence] #this takes care of looping or excessive time spent - break if you loop (40 s should be definitely enough to move on)
            break
        if parser_sim.current_event.action == "KEY PRESSED: SPACE":
            extra_rule_time = parser.model_parameters["latency_factor"]*np.exp(-parser.model_parameters["latency_exponent"]* np.mean(activations)/10)
            # two things play a role - number of matching features; fan of each matching feature; store these two separately for later explorations
            if len(spr_times) not in tobe_removed:
                spr_times.append(parser_sim.show_time() + extra_rule_time - last_time)
            else:
                tobe_removed.remove(len(spr_times))
            last_time = parser_sim.show_time()
            word_parsed += 1
        if word_parsed >= len(sentence)-2: #we ignore the last two words
            if len(spr_times) not in tobe_removed:
                spr_times.append(parser_sim.show_time() - last_time)
            break
        #this below implements carrying out an action

        if re.search("^RULE FIRED: recall action", parser_sim.current_event.action) or\
                                re.search("^RULE FIRED: move to last action", parser_sim.current_event.action):
            postulated_gaps, reduced_unary = 0, 0
            postulate_gaps, reduce_unary = True, True
            parser_sim.steps(2) #exactly enough steps to make imaginal full
            if prints:
                print(parser.goals["imaginal"])
            #add new word to the list of used words
            built_constituents.append((Tree(str(parser.goals["imaginal"].copy().pop().TREE0_LABEL), (str(parser.goals["imaginal"].copy().pop().TREE0_HEAD),)), (None, str(parser.goals["imaginal"].copy().pop().TREE0_HEAD))))
            built_constituents_reanalysis = built_constituents.copy()
            parser.goals["imaginal_reanalysis"].add(parser.goals["imaginal"].copy().pop())
            recently_retrieved = set()

            #set retrieve_wh to None or, if the reanalysis already postulated a gap, to "yes"
            retrieve_wh = retrieve_wh_reanalysis
            retrieve_wh_reanalysis = None
            activations, agreeing_actions, matching_fs, total_fan = [], [], [], [] #collects total activation of rules, agreeing_actions... (used to find out what plays a role in syntactic parsing for RTs)

            #antecedent_carried temporarily updated in the blind analysis; we will record the original position, which is the non-temporary one, in antecedent_carried_origo and re-use it after the blind analysis; what_antecedent_carried - specifies the category of antecedent
            antecedent_carried_origo = antecedent_carried

            first_action = True

            reanalysis_list.append("no") #by dft no reanalysis recorded
            #this loop for actual blind analysis
            while True:
                parser_retrievals, number_of_agreeing_actions, number_of_matching_fs, fan_size = ut.recall_action(blind_actions, parser.goals["imaginal"], parser.goals["word_info"], recently_retrieved, built_constituents, word_freq, label_freq, prints=False, strength_of_association=strength_of_association, postulate_gaps=postulate_gaps, reduce_unary=reduce_unary, blind={"WORD_NEXT0_LEX", "WORD_NEXT0_POS"})
                activations.append(parser_retrievals[0])
                agreeing_actions.append(number_of_agreeing_actions)
                matching_fs.append(number_of_matching_fs)
                total_fan.append(fan_size)

                if first_action:
                    actions_list.append(str(parser_retrievals[1]["action"]))
                    first_action = False

                ut.collect_parse(parser_retrievals[1], built_constituents)
                tree0_label = built_constituents[-1][0].label()
                tree1_label = built_constituents[-2][0].label()
                tree2_label = built_constituents[-3][0].label()
                tree3_label = built_constituents[-4][0].label()
                children = {"".join(["tree", str(x)]): ["NOPOS", "NOPOS"] for x in range(4)}
                for x, subtree in enumerate(built_constituents[-1][0]):
                    if isinstance(subtree, Tree) and subtree.label() != ut.EMPTY:
                        children["tree0"][x] = subtree.label()
                if re.search("_BAR", children["tree0"][1]):
                    if built_constituents[-1][0][1][1].label() == ut.EMPTY or re.search("_BAR", built_constituents[-1][0][1][1].label()):
                        children["tree0"][1] = built_constituents[-1][0][1][0].label()
                    else:
                        children["tree0"][1] = built_constituents[-1][0][1][1].label()
                for x, subtree in enumerate(built_constituents[-2][0]):
                    if isinstance(subtree, Tree) and subtree.label() != ut.EMPTY:
                        children["tree1"][x] = subtree.label()
                if re.search("_BAR", children["tree1"][1]):
                    if built_constituents[-2][0][1][1].label() == ut.EMPTY or re.search("_BAR", built_constituents[-2][0][1][1].label()):
                        children["tree1"][1] = built_constituents[-2][0][1][0].label()
                    else:
                        children["tree1"][1] = built_constituents[-2][0][1][1].label()

                # block looping through reduce_unary (at most 2 reduce_unary allowed)
                if parser_retrievals[1] and parser_retrievals[1]["action"] == 'reduce_unary':
                    reduced_unary += 1
                    if reduced_unary == 2:
                        reduce_unary = False
                        reduced_unary = 0
                else:
                    reduced_unary = 0
                    reduce_unary = True
                if parser_retrievals[1] and parser_retrievals[1]["action"] == 'postulate_gap':
                    if antecedent_carried == "YES" and syntactic and re.search("t", str(parser_retrievals[1]["action_result_label"][0])):
                        retrieve_wh = "yes"
                    if re.search("t", str(parser_retrievals[1]["action_result_label"][0])):
                        antecedent_carried = "NO"
                    #at most 3 gaps allowed
                    if postulated_gaps > 1:
                        postulate_gaps = False
                    postulated_gaps += 1
                    ci = parser.goals["imaginal"].pop()

                    string="""
    isa             action_chunk
    WORD_NEXT0_LEX   """+'"'+str(ci.WORD_NEXT0_LEX)+'"'+"""
    WORD_NEXT0_POS   '"""+str(ci.WORD_NEXT0_POS)+"""'
    ANTECEDENT_CARRIED      """+antecedent_carried+"""
    TREE0_HEAD       """+'"'+str(parser_retrievals[1]["action_result_label"][0])+'"'+"""
    TREE0_LEFTCHILD    """+children["tree0"][0]+"""
    TREE0_RIGHTCHILD    """+children["tree0"][1]+"""
    TREE0_LABEL       '-NONE-'
    TREE1_LEFTCHILD    """+children["tree1"][0]+"""
    TREE1_RIGHTCHILD    """+children["tree1"][1]+"""
    TREE0_HEADPOS     """+str(built_constituents[-1][1][0])+"""
    TREE1_LABEL     """+'"'+tree1_label+'"'+"""
    TREE1_HEADPOS     """+str(built_constituents[-2][1][0])+"""
    TREE1_HEAD     """+'"'+str(built_constituents[-2][1][1])+'"'+"""
    TREE2_LABEL     """+'"'+tree2_label+'"'+"""
    TREE2_HEADPOS     """+str(built_constituents[-3][1][0])+"""
    TREE2_HEAD     """+'"'+str(built_constituents[-3][1][1])+'"'+"""
    TREE3_LABEL     """+'"'+tree3_label+'"'+"""
    TREE3_HEAD     """+'"'+str(built_constituents[-4][1][1])+'"'+"""
    ACTION_PREV     """+str(parser_retrievals[1]["action"])
                    parser.goals["imaginal"].add(actr.chunkstring(string=string))
                    parser.goals["word_info"].add(actr.chunkstring(string="""
                    isa         word
                    form       '"""+str(parser_retrievals[1]["action_result_label"][0])+"""'
                    cat         '-NONE-'"""))

                elif parser_retrievals[1]:
                    ci = parser.goals["imaginal"].pop()

                    string="""
    isa             action_chunk
    WORD_NEXT0_LEX   """+'"'+str(ci.WORD_NEXT0_LEX)+'"'+"""
    WORD_NEXT0_POS   '"""+str(ci.WORD_NEXT0_POS)+"""'
    ANTECEDENT_CARRIED      """+antecedent_carried+"""
    TREE0_LABEL     """+'"'+str(built_constituents[-1][0].label())+'"'+"""
    TREE0_HEADPOS     """+str(built_constituents[-1][1][0])+"""
    TREE0_HEAD     """+'"'+str(built_constituents[-1][1][1])+'"'+"""
    TREE0_LEFTCHILD    """+children["tree0"][0]+"""
    TREE0_RIGHTCHILD    """+children["tree0"][1]+"""
    TREE1_LABEL     """+'"'+tree1_label+'"'+"""
    TREE1_HEADPOS     """+str(built_constituents[-2][1][0])+"""
    TREE1_HEAD     """+'"'+str(built_constituents[-2][1][1])+'"'+"""
    TREE1_LEFTCHILD    """+children["tree1"][0]+"""
    TREE1_RIGHTCHILD    """+children["tree1"][1]+"""
    TREE2_LABEL     """+'"'+tree2_label+'"'+"""
    TREE2_HEADPOS     """+str(built_constituents[-3][1][0])+"""
    TREE2_HEAD     """+'"'+str(built_constituents[-3][1][1])+'"'+"""
    TREE3_LABEL     """+'"'+tree3_label+'"'+"""
    TREE3_HEAD     """+'"'+str(built_constituents[-4][1][1])+'"'+"""
    ACTION_PREV     """+str(parser_retrievals[1]["action"])
                    parser.goals["imaginal"].add(actr.chunkstring(string=string))
                else:
                    break
                if parser_retrievals[1]["action"] == 'shift':
                    #sometimes the parser would stop at BAR and shift; in reality, this is not possible since BARs are artificial categories
                    if re.search("_BAR", built_constituents[-1][0].label()):
                        built_constituents[-1][0].set_label(re.split("_BAR", built_constituents[-1][0].label())[0])
                    ci = parser.goals["imaginal"].pop()

                    string = """
    isa             action_chunk
    TREE1_LABEL     """+'"'+tree0_label+'"'+"""
    TREE1_HEADPOS     """+str(built_constituents[-1][1][0])+"""
    TREE1_HEAD     """+'"'+str(built_constituents[-1][1][1])+'"'+"""
    TREE1_LEFTCHILD    """+children["tree0"][0]+"""
    TREE1_RIGHTCHILD    """+children["tree0"][1]+"""
    TREE2_LABEL     """+'"'+tree1_label+'"'+"""
    TREE2_HEADPOS     """+str(built_constituents[-2][1][0])+"""
    TREE2_HEAD     """+'"'+str(built_constituents[-2][1][1])+'"'+"""
    TREE3_LABEL     """+'"'+tree2_label+'"'+"""
    TREE3_HEAD     """+'"'+str(built_constituents[-3][1][1])+'"'+"""
    ANTECEDENT_CARRIED      """+antecedent_carried+"""
    ACTION_PREV     """+str(parser_retrievals[1]["action"])
                    parser.goals["imaginal"].add(actr.chunkstring(string=string))
                    break

            postulated_gaps, reduced_unary = 0, 0
            postulate_gaps, reduce_unary = True, True
            
            antecedent_carried = antecedent_carried_origo
            #this loop for potential reanalysis
            while True:
                parser_retrievals, _, _, _ = ut.recall_action(actions, parser.goals["imaginal_reanalysis"], parser.goals["word_info"], recently_retrieved, built_constituents_reanalysis, word_freq, label_freq, prints=False, strength_of_association=strength_of_association_reanalysis, number_retrieved=3, postulate_gaps=postulate_gaps, reduce_unary=reduce_unary, blind={})
                ut.collect_parse(parser_retrievals[1], built_constituents_reanalysis)
                tree0_label = built_constituents_reanalysis[-1][0].label()
                tree1_label = built_constituents_reanalysis[-2][0].label()
                tree2_label = built_constituents_reanalysis[-3][0].label()
                tree3_label = built_constituents_reanalysis[-4][0].label()
                children = {"".join(["tree", str(x)]): ["NOPOS", "NOPOS"] for x in range(4)}
                for x, subtree in enumerate(built_constituents[-1][0]):
                    if isinstance(subtree, Tree) and subtree.label() != ut.EMPTY:
                        children["tree0"][x] = subtree.label()
                if re.search("_BAR", children["tree0"][1]):
                    if built_constituents[-1][0][1][1].label() == ut.EMPTY or re.search("_BAR", built_constituents[-1][0][1][1].label()):
                        children["tree0"][1] = built_constituents[-1][0][1][0].label()
                    else:
                        children["tree0"][1] = built_constituents[-1][0][1][1].label()
                for x, subtree in enumerate(built_constituents[-2][0]):
                    if isinstance(subtree, Tree) and subtree.label() != ut.EMPTY:
                        children["tree1"][x] = subtree.label()
                if re.search("_BAR", children["tree1"][1]):
                    if built_constituents[-2][0][1][1].label() == ut.EMPTY or re.search("_BAR", built_constituents[-2][0][1][1].label()):
                        children["tree1"][1] = built_constituents[-2][0][1][0].label()
                    else:
                        children["tree1"][1] = built_constituents[-2][0][1][1].label()

                if re.search("-TPC", tree0_label) or (re.search("^W", tree0_label)):
                    antecedent_carried = "YES"
                    what_antecedent_carried = str(tree0_label)

                # block looping through reduce_unary (at most 2 reduce_unary allowed)
                if parser_retrievals[1] and parser_retrievals[1]["action"] == 'reduce_unary':
                    reduced_unary += 1
                    if reduced_unary == 2:
                        reduce_unary = False
                        reduced_unary = 0
                else:
                    reduced_unary = 0
                    reduce_unary = True
                if parser_retrievals[1] and parser_retrievals[1]["action"] == 'postulate_gap':
                    if antecedent_carried_origo == "YES" and syntactic and re.search("t", str(parser_retrievals[1]["action_result_label"][0])) and retrieve_wh != "yes":
                        retrieve_wh_reanalysis = "yes" #record that based on the upcoming word info, trace should be postulated; only if the original structure did not postulate it
                    if re.search("t", str(parser_retrievals[1]["action_result_label"][0])):
                        antecedent_carried = "NO"
                    #at most 3 gaps allowed
                    if postulated_gaps > 1:
                        postulate_gaps = False
                    postulated_gaps += 1
                    ci = parser.goals["imaginal_reanalysis"].pop()
                    parser.decmem.add(ci, time=parser_sim.show_time())

                    string="""
    isa             action_chunk
    WORD_NEXT0_LEX   """+'"'+str(ci.WORD_NEXT0_LEX)+'"'+"""
    WORD_NEXT0_POS   """+'"'+str(ci.WORD_NEXT0_POS)+'"'+"""
    ANTECEDENT_CARRIED      """+antecedent_carried+"""
    TREE0_HEAD       """+'"'+str(parser_retrievals[1]["action_result_label"][0])+'"'+"""
    TREE0_LABEL       '-NONE-'
    TREE0_HEADPOS     """+str(built_constituents_reanalysis[-1][1][0])+"""
    TREE0_LEFTCHILD    """+children["tree0"][0]+"""
    TREE0_RIGHTCHILD    """+children["tree0"][1]+"""
    TREE1_LABEL     """+'"'+tree1_label+'"'+"""
    TREE1_HEADPOS     """+str(built_constituents_reanalysis[-2][1][0])+"""
    TREE1_HEAD     """+'"'+str(built_constituents_reanalysis[-2][1][1])+'"'+"""
    TREE1_LEFTCHILD    """+children["tree1"][0]+"""
    TREE1_RIGHTCHILD    """+children["tree1"][1]+"""
    TREE2_LABEL     """+'"'+tree2_label+'"'+"""
    TREE2_HEADPOS     """+str(built_constituents_reanalysis[-3][1][0])+"""
    TREE2_HEAD     """+'"'+str(built_constituents_reanalysis[-3][1][1])+'"'+"""
    TREE3_LABEL     """+'"'+tree3_label+'"'+"""
    TREE3_HEAD     """+'"'+str(built_constituents_reanalysis[-4][1][1])+'"'+"""
    ACTION_PREV     """+str(parser_retrievals[1]["action"])
                    parser.goals["imaginal_reanalysis"].add(actr.chunkstring(string=string))
                    parser.goals["word_info"].add(actr.chunkstring(string="""
                    isa         word
                    form       '"""+str(parser_retrievals[1]["action_result_label"][0])+"""'
                    cat         '-NONE-'"""))

                elif parser_retrievals[1]:
                    ci = parser.goals["imaginal_reanalysis"].pop()
                    parser.decmem.add(ci, time=parser_sim.show_time())

                    string="""
    isa             action_chunk
    WORD_NEXT0_LEX   """+'"'+str(ci.WORD_NEXT0_LEX)+'"'+"""
    WORD_NEXT0_POS   """+'"'+str(ci.WORD_NEXT0_POS)+'"'+"""
    ANTECEDENT_CARRIED      """+antecedent_carried+"""
    TREE0_LABEL     """+'"'+str(built_constituents_reanalysis[-1][0].label())+'"'+"""
    TREE0_HEADPOS     """+str(built_constituents_reanalysis[-1][1][0])+"""
    TREE0_HEAD     """+'"'+str(built_constituents_reanalysis[-1][1][1])+'"'+"""
    TREE0_LEFTCHILD    """+children["tree0"][0]+"""
    TREE0_RIGHTCHILD    """+children["tree0"][1]+"""
    TREE1_LABEL     """+'"'+tree1_label+'"'+"""
    TREE1_HEADPOS     """+str(built_constituents_reanalysis[-2][1][0])+"""
    TREE1_HEAD     """+'"'+str(built_constituents_reanalysis[-2][1][1])+'"'+"""
    TREE1_LEFTCHILD    """+children["tree1"][0]+"""
    TREE1_RIGHTCHILD    """+children["tree1"][1]+"""
    TREE2_LABEL     """+'"'+tree2_label+'"'+"""
    TREE2_HEADPOS     """+str(built_constituents_reanalysis[-3][1][0])+"""
    TREE2_HEAD    """+'"'+str(built_constituents_reanalysis[-3][1][1])+'"'+"""
    TREE3_LABEL     """+'"'+tree3_label+'"'+"""
    TREE3_HEAD     """+'"'+str(built_constituents_reanalysis[-4][1][1])+'"'+"""
    ACTION_PREV     """+str(parser_retrievals[1]["action"])
                    parser.goals["imaginal_reanalysis"].add(actr.chunkstring(string=string))
                else:
                    break
                if parser_retrievals[1]["action"] == 'shift':
                    #sometimes the parser would stop at BAR and shift; in reality, this is not possible since BARs are artificial categories
                    if re.search("_BAR", built_constituents_reanalysis[-1][0].label()):
                        built_constituents_reanalysis[-1][0].set_label(re.split("_BAR", built_constituents_reanalysis[-1][0].label())[0])
                    ci = parser.goals["imaginal_reanalysis"].pop()
                    parser.decmem.add(ci, time=parser_sim.show_time())
                    #built constituents have head info; if it is not present, use the info from imaginal_reanalysis (stores head info for terminal nodes)

                    string = """
    isa             action_chunk
    TREE1_LABEL     """+'"'+tree0_label+'"'+"""
    TREE1_HEADPOS     """+str(built_constituents_reanalysis[-1][1][0])+"""
    TREE1_HEAD     """+'"'+str(built_constituents_reanalysis[-1][1][1])+'"'+"""
    TREE1_LEFTCHILD    """+children["tree0"][0]+"""
    TREE1_RIGHTCHILD    """+children["tree0"][1]+"""
    TREE2_LABEL     """+'"'+tree1_label+'"'+"""
    TREE2_HEADPOS     """+str(built_constituents_reanalysis[-2][1][0])+"""
    TREE2_HEAD     """+'"'+str(built_constituents_reanalysis[-2][1][1])+'"'+"""
    TREE3_LABEL     """+'"'+tree2_label+'"'+"""
    TREE3_HEAD     """+'"'+str(built_constituents_reanalysis[-3][1][1])+'"'+"""
    ANTECEDENT_CARRIED      """+antecedent_carried+"""
    ACTION_PREV     """+str(parser_retrievals[1]["action"])
                    parser.goals["imaginal_reanalysis"].add(actr.chunkstring(string=string))
                    break

            cg = parser.goals["g"].pop()
            parser.goals["g"].add(actr.chunkstring(string="""
    isa             reading
    position    """+str(cg.position)+"""
    reanalysis      None
    retrieve_wh     """+str(retrieve_wh)+"""
    what_retrieve     """+str(what_antecedent_carried)#used only for recall of category
    +"""
    state           finished_recall"""))
            if built_constituents != built_constituents_reanalysis:
                if reanalysis and len(built_constituents) != len(built_constituents_reanalysis):
                    #mark that the reanalysis should take place
                    parser.goals["g"].add(actr.chunkstring(string="""
    isa             reading
    position    """+str(cg.position)+"""
    reanalysis      yes
    retrieve_wh     """+str(retrieve_wh)+"""
    what_retrieve     """+str(what_antecedent_carried)#used only for recall of category
    +"""
    state           finished_recall"""))
                    reanalysis_list[-1] = "yes"
                    if prints:
                        original_tree = Tree("X", next(zip(*built_constituents[3:])))
                        print("DRAWING TREE TO BE REANALYSED")
                        print("********************************")
                        original_tree.draw()
                built_constituents = built_constituents_reanalysis.copy()
                parser.goals["imaginal"].add(parser.goals["imaginal_reanalysis"].copy().pop())

            final_tree = Tree("X", next(zip(*built_constituents[3:])))
            if word_parsed not in tobe_removed:
                activations_list.append(np.mean(activations)/10)
                wh_gaps_list.append(str(retrieve_wh))
                agreeing_actions_list.append(np.mean(agreeing_actions))
                matching_fs_list.append(np.mean(matching_fs))
                total_fan_list.append(np.mean(total_fan))
                words_list.append(sentence[word_parsed])

            if prints:
                print("DRAWING TREE")
                print("********************************")
                # print(final_tree)
                # final_tree.pretty_print()
                final_tree.draw()

    final_times = [x for x in spr_times if x not in tobe_removed]
    final_times = final_times[:-1] #remove the last time (period)
    if prints:
        print("FINAL TIMES")
        print(final_times)
        print(activations_list)
        # with open('parses/parse_trees.txt', 'a+') as f:
        #     f.write(str(final_tree) + "\n")
        #     f.write("\n")
    return words_list, final_times, activations_list, wh_gaps_list, reanalysis_list, agreeing_actions_list, matching_fs_list, total_fan_list
    
if __name__ == "__main__":
    stimuli_csv = ut.load_file(SENTENCES, sep=",") #sentences with frequencies
    words = ut.load_file(WORDS, sep="\t")
    labels = ut.load_file(LABELS, sep="\t")
    actions = ut.load_file(ACTIONS, sep="\t")
    blind_actions = ut.load_file(BLIND_ACTIONS, sep="\t")
    DM = parser.decmem.copy()

    #prepare dictionaries to calculate spreading activation
    word_freq = {k: sum(g["FREQ"].tolist()) for k,g in words.groupby("WORD")} #this method sums up frequencies of words across all POS
    label_freq = labels.set_index('LABEL')['FREQ'].to_dict()
    
    # Save the respective activations
    total = {"activation": [], "item": [], "position": [], "word": [], "retrieve_wh": [], "reanalysis": [], "agreeing_actions": [], "matching_fs": [], "fan_size": [], "condition": []}

    for condition in (set(stimuli_csv.label.to_numpy())):
        for sent_nr in range(1, 17):
            sent_nr = str(sent_nr)
            print(condition, sent_nr)
            subset_stimuli = stimuli_csv[stimuli_csv.label.isin([condition]) & stimuli_csv.item.isin([sent_nr])]
            words_list, final_times, acts, whs, reanalysis_list, agreeing_actions, matching_fs, fan_sizes = read(parser, sentence=subset_stimuli.word.tolist(), pos=subset_stimuli.function.tolist(), actions=actions, blind_actions=blind_actions, word_freq=word_freq, label_freq=label_freq, decmem=DM, lexical=False, syntactic=True, visual=False, reanalysis=True, prints=True) # set Prints to false if you do not want to watch tree-building
            total["word"] += words_list
            total["activation"] += acts
            total["position"] += [i+1 for i in range(len(words_list))]
            total["condition"] += [condition for _ in range(len(words_list))]
            total["item"] += [sent_nr for _ in range(len(words_list))]
            total["retrieve_wh"] += whs
            total["reanalysis"] += reanalysis_list
            total["agreeing_actions"] += agreeing_actions
            total["matching_fs"] += matching_fs
            total["fan_size"] += fan_sizes

            for key in total:
                print(key)
                print(len(total[key]))
    
    activations_df = pd.DataFrame.from_dict(total)
    activations_df.to_csv("activations_"+SENTENCES, sep="\t", encoding="utf-8", index=False)

