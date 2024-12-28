# Importing necessary libraries for data manipulation, NLP processing, and mathematical operations
from collections import Counter
from nltk.data import load
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import pandas as pd
import nltk
import pickle
import random
import csv

# Downloading the treebank dataset and POS tagset from the NLTK library
nltk.download('treebank')
nltk.download('tagsets')

# Loading the universal part of speech tags provided by the NLTK library
tagdict = load('help/tagsets/upenn_tagset.pickle')
TAGS = list(tagdict.keys())
TAGS.append("unk") # Adding an "UNK" (unknown) tag for words that don't fit into known categories

# Function to read data from a file and format it for processing
def read(fileName):
    finalList = []
    tmp = []
    # Open the file with the given fileName
    with open("data/"+fileName) as tsv:
        # Reading each line of the file
        for line in csv.reader(tsv, dialect="excel-tab"):
            if len(line) == 0:  # New Line
                if fileName != "test" and len(tmp) > 0 and tmp[-1][1] != ".":
                    # Ensure sentences in training and development sets end with a period
                    tmp.append(tuple(["","."]))
                finalList.append(tmp)
                tmp = []
            else:
                if fileName == "test":
                    tmp.append(tuple([line[1]]))
                else:
                    # Train or Dev
                    tmp.append(tuple([line[1], line[2]]))
    return finalList

def cSentences(l):
    finalList = []
    for i in l:
        for j in i:
            finalList.append(j)
    return finalList

trainL = read("train")
testL = read("test")
combData = cSentences(trainL)

TRAIN_TAGS = [tag[1] for tag in combData]
TRAIN_WORDS = [tag[0] for tag in combData]

unknown = 0
final = []

for index, tag in enumerate(TRAIN_TAGS):
    if tag not in set(TAGS):
        unknown += 1
        TRAIN_TAGS[index] = "UNK"

filehandler = open("t_prob.obj","rb")
tags_transition_prob = pickle.load(filehandler)
filehandler.close()

filehandler = open("e_prob.obj","rb")
emission_dict = pickle.load(filehandler)
filehandler.close()

initial_prob = {}
initial_total = 0
for d in trainL:
    d = d[0]
    if d[1] not in initial_prob:
        initial_prob[d[1]] = 1
    else:
        tmp = initial_prob[d[1]]
        initial_prob[d[1]] = tmp + 1
    initial_total += 1

for s in testL:
    sentence_words = [t[0] for t in s]
    final_prob_matrix = [[0 for _ in range(len(TAGS))] for _ in range(len(sentence_words))]
    tag_ind_matrix = [[0 for _ in range(len(TAGS))] for _ in range(len(sentence_words)-1)]

    period_ind = TAGS.index(".")
    for tag_ind, tag in enumerate(TAGS):

        word = sentence_words[0]

        transition_prob = 0
        if tag in initial_prob:
            transition_prob = initial_prob[tag]/initial_total
        emission_prob = 0.003008641237291414
        if word in emission_dict:
            emission_prob = emission_dict[word][tag_ind]
        final_prob_matrix[0][tag_ind] = emission_prob * transition_prob

    for word_ind in range(1, len(sentence_words)):
        word = sentence_words[word_ind]
        for tag_ind, curr_tag in enumerate(TAGS):
            max_transit_prob = 0
            max_transition_ind = 45
            emission_prob = 0.003008641237291414
            if word in emission_dict:
                emission_prob = emission_dict[word][tag_ind]

            # Calculating the best previous state and transition probability
            for prev_ind, prev_tag in enumerate(TAGS):
                transition_prob = tags_transition_prob[prev_ind][tag_ind]
                prev_prob = final_prob_matrix[word_ind-1][prev_ind]

                tmp_prob = prev_prob * transition_prob
                if max_transit_prob < tmp_prob:
                    max_transit_prob = tmp_prob
                    max_transition_ind = prev_ind

            # Updating matrices with the best probabilities and backpointers
            final_prob_matrix[word_ind][tag_ind] = emission_prob * max_transit_prob
            tag_ind_matrix[word_ind-1][tag_ind] = max_transition_ind

    # Backtracking to find the best path of tags
    viterbi_predicted_tags = []
    max_value = -1
    max_ind = 45
    for index, value in enumerate(final_prob_matrix[-1]):
        if value > max_value:
            max_value = value
            max_ind = index
    viterbi_predicted_tags.append(TAGS[max_ind])

    prev_index = max_ind
    for n in range(len(sentence_words)-2, -1, -1):
        index = tag_ind_matrix[n][prev_index]
        viterbi_predicted_tags.append(TAGS[index])
        prev_index = index

    viterbi_predicted_tags.reverse()
    for word_ind, word in enumerate(sentence_words):
        final_tag = viterbi_predicted_tags[word_ind]
        final.append(str(word_ind+1)+"\t"+word+"\t"+final_tag)
    final.append("")

with open("viterbi.out", "w") as ofile:
    for line in final:
        ofile.write(f'{line}\n')
