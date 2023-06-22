#!/usr/bin/env python
import json
from itertools import chain
import pickle
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    print(document.text.split())
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')  # 构建邻接矩阵。

    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    #matrix[child.i][token.i] = 1
    return matrix

#################################################################################
# SENTENCES

ABAM_DATA_SENTENCES_without_sentences = pd.read_csv(
        '../data/ABAM_DATA_SENTENCES.tsv',
        sep='\t')
print(len(ABAM_DATA_SENTENCES_without_sentences))
a1 = np.array(ABAM_DATA_SENTENCES_without_sentences)
print(a1[1])
ABAM_SENTENCES = pd.read_csv(
        '../data/ABAM_SENTENCES.tsv',
        sep='\t')
print(len(ABAM_SENTENCES))
a2 = np.array(ABAM_SENTENCES)
print(a2[1])
hash_dict={}
for i in a2:
    hash_dict[i[0]]=i[1]

fout = open('../data/spacy.txt'+'.tree', 'wb')
idx2graph = {}

data_dicts=json.load(open('../data/data_dict.json','r',encoding='utf-8'))
data_dicts_bert={}
#topic	sentence_hash	sentence	stance	aspect	inner	cross
for k,v in data_dicts.items():
    data_dict=v
    sentence = data_dict['tokenized_sentence_spacy'].lower()
    pos = data_dict['tokenized_sentence_spacy_pos']
    #sentence = data_dict['tokenized_sentence_spacy'].lower()


    as_label=[int(label[0]) for label in data_dict['tokenized_sentence_spacy_as_labels'].split()]
    am_label=[label[0] for label in data_dict['tokenized_sentence_spacy_labels'].split()]
    adj_matrix = dependency_adj_matrix(sentence)
    print(adj_matrix)
    print(as_label)
    print(am_label)
    print(pos)
    idx2graph[int(k)]=adj_matrix



    data_dicts_bert[k]=data_dict
pickle.dump(idx2graph, fout)
fout.close()
with open('../data/data_dict_bert_pieces2word2.json', 'w') as my_file:
    json.dump(data_dicts_bert, my_file, sort_keys=True, indent=4, separators=(',', ': '))

assert len(ABAM_DATA_SENTENCES_without_sentences)==len(ABAM_SENTENCES)

ABAM_DATA_SENTENCES = pd.merge(
        ABAM_DATA_SENTENCES_without_sentences[['topic', 'sentence_hash', 'stance', 'aspect', 'inner', 'cross']],
        ABAM_SENTENCES,
        on='sentence_hash')
ABAM_DATA_SENTENCES = ABAM_DATA_SENTENCES[['topic', 'sentence_hash', 'sentence', 'stance', 'aspect', 'inner', 'cross']]
print(len(ABAM_DATA_SENTENCES))
a3 = np.array(ABAM_DATA_SENTENCES)
print(a3[1])

print(ABAM_DATA_SENTENCES[['topic', 'sentence_hash']].groupby('topic').count())

#################################################################################
# SEGMENTS

ABAM_DATA_SEGMENTS_without_segments = pd.read_csv(
        '../data/ABAM_DATA_SEGMENTS.tsv',
        sep='\t')
print(len(ABAM_DATA_SEGMENTS_without_segments))
a4 = np.array(ABAM_DATA_SEGMENTS_without_segments)
print(a4[1])

ABAM_SEGMENTS = pd.read_csv(
        '../data/ABAM_SEGMENTS.tsv',
        sep='\t')
print(len(ABAM_SEGMENTS))
a5 = np.array(ABAM_SEGMENTS)
print(a5[1])

assert len(ABAM_DATA_SEGMENTS_without_segments)==len(ABAM_SEGMENTS)

ABAM_DATA_SEGMENTS = pd.merge(
        ABAM_DATA_SEGMENTS_without_segments[['topic', 'sentence_hash', 'segment_count', 'segment_hash', 'stance', 'aspect', 'inner', 'cross']],
        ABAM_SEGMENTS, on='segment_hash')
ABAM_DATA_SEGMENTS = ABAM_DATA_SEGMENTS[['topic', 'sentence_hash', 'segment_count', 'segment_hash', 'segment', 'stance', 'aspect', 'inner', 'cross']]
print(len(ABAM_DATA_SEGMENTS))
a6 = np.array(ABAM_DATA_SEGMENTS)
print(a6[1])

print(ABAM_DATA_SEGMENTS[['topic', 'segment_hash']].groupby('topic').count())
