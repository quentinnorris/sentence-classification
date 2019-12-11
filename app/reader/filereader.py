from __future__ import division
from __future__ import print_function

import os

import numpy as np


def read_glove_vectors(glove_vector_path):
    '''Method to read glove vectors and return an embedding dict.'''
    embeddings_index = {}
    #with open(glove_vector_path, 'r')
    with open(glove_vector_path, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs[:]
    return embeddings_index

def read_input_data(input_data_path):
    '''Method to read data from input_data_path'''
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    # AREA TO COMMENT OUT/COMMENT IN WHICHEVER TYPE OF DATA YOU WANT TO UTILIZE


    #texts = list(open(os.path.join(input_data_path, "input.txt"), "r").readlines()) #SAMPLE DATASET
    texts = list(open(os.path.join(input_data_path, "input_MR.txt"), encoding='latin-1')) # MR DATA SET
    with open(os.path.join(input_data_path, "label_MR.txt"), encoding='latin-1') as label_f: # MR LABELS
    #texts = list(open(os.path.join(input_data_path, "input_subj.txt"), "r").readlines()) # SUBJ DATA SET
    #with open(os.path.join(input_data_path, "label_subj.txt"), 'r') as label_f: #SUBJ LABELS
    #texts = list(open(os.path.join(input_data_path, "input_trec.txt"), "r").readlines()) # TREC DATA SET
    #with open(os.path.join(input_data_path, "label_trec.txt"), 'r') as label_f: #TREC LABELS
    # texts = list(open(os.path.join(input_data_path, "input_sst2.txt"), "r").readlines()) # SST2 DATA SET
    # with open(os.path.join(input_data_path, "label_sst2.txt"), 'r') as label_f: #SST2 LABELS
        largest_label_id = 0
        for line in label_f:
            label = str(line.strip())
            if label not in labels_index:
                labels_index[label] = largest_label_id
                largest_label_id += 1
            labels.append(labels_index[label])

    return texts, labels_index, labels
