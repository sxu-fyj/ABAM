import json
from itertools import chain
import pickle
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")
# from transformers import BertTokenizer
#
# tokenizer = BertTokenizer.from_pretrained('../model_base/')


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
                    matrix[child.i][token.i] = 1
    return matrix

#################################################################################
# SENTENCES
data_dicts={}

with open('../AURC/AURC_DATA_dict.json', 'r') as my_file:
    AURC_DATA_dict = json.load(my_file)
count= 0
for topic, AD in AURC_DATA_dict.items():
    for ad in AD:
        data_dict={}
        data_dict['sentence']=ad['sentence']
        data_dict['topic']=topic
        data_dict['in-domain']=ad['In-Domain']
        data_dict['cross-domain']=ad['Cross-Domain']

        data_dict['tokenized_sentence_spacy_labels'] = ad['tokenized_sentence_spacy_labels']
        data_dict['tokenized_sentence_spacy'] = ad['tokenized_sentence_spacy']
        data_dict['tokenized_sentence_bert_labels'] = ad['tokenized_sentence_bert_labels']
        data_dict['tokenized_sentence_bert'] = ad['tokenized_sentence_bert']
        data_dicts[count] = data_dict
        count += 1

fout = open('../AURC/spacy.txt'+'.graph', 'wb')
idx2graph = {}
data_dicts_bert={}
#topic	sentence_hash	sentence	stance	aspect	inner	cross
for k,v in data_dicts.items():
    data_dict=v
    sentence = data_dict['tokenized_sentence_spacy'].lower()

    adj_matrix = dependency_adj_matrix(sentence)
    print(len(sentence.split()))
    print(len(adj_matrix))
    assert len(sentence.split())==len(adj_matrix)
    idx2graph[int(k)]=adj_matrix

    data_dicts_bert[k]=data_dict
pickle.dump(idx2graph, fout)
fout.close()
with open('../AURC/data_dict_bert.json', 'w') as my_file:
    json.dump(data_dicts_bert, my_file, sort_keys=True, indent=4, separators=(',', ': '))
