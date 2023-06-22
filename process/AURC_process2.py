import json
from itertools import chain
import pickle as pkl
import pandas as pd
import numpy as np
# import spacy
# nlp = spacy.load("en_core_web_sm")
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../model_base/')


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
# data_dicts={}
#
# with open('../AURC/AURC_DATA_dict.json', 'r') as my_file:
#     AURC_DATA_dict = json.load(my_file)
# count= 0
# for topic, AD in AURC_DATA_dict.items():
#     for ad in AD:
#         data_dict={}
#         data_dict['sentence']=ad['sentence']
#         data_dict['in-domain']=ad['In-Domain']
#         data_dict['cross-domain']=ad['Cross-Domain']
#
#         data_dict['tokenized_sentence_spacy_labels'] = ad['tokenized_sentence_spacy_labels']
#         data_dict['tokenized_sentence_spacy'] = ad['tokenized_sentence_spacy']
#         data_dict['tokenized_sentence_bert_labels'] = ad['tokenized_sentence_bert_labels']
#         data_dict['tokenized_sentence_bert'] = ad['tokenized_sentence_bert']
#         data_dicts[count] = data_dict
#         count += 1

with open('../AURC/spacy.txt.graph', 'rb') as fr:
    graph = pkl.load(fr)
    print(1)

idx2graph = {}
data_dicts_bert={}
pieces2word={}
data_dicts=json.load(open('../AURC/data_dict_bert.json','r',encoding='utf-8'))

#topic	sentence_hash	sentence	stance	aspect	inner	cross
for k,v in data_dicts.items():
    data_dict=v
    sentence = data_dict['tokenized_sentence_spacy'].lower()
    #sentence = data_dict['tokenized_sentence_spacy'].lower()


    idsw = tokenizer.encode(sentence, add_special_tokens=False)
    tokenized_sentence_bert = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=False))
    length = len(sentence.split())

    tokens = [tokenizer.tokenize(word) for word in sentence.split()]
    pieces = [piece for pieces in tokens for piece in pieces]
    _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
    _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
    _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)

    if tokenizer is not None:
        start = 0
        for i, pieces in enumerate(tokens):
            if len(pieces) == 0:
                continue
            pieces = list(range(start, start + len(pieces)))
            _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
            start += len(pieces)
    assert len(graph[int(k)]) == len(_pieces2word)
    pieces2word[int(k)]=_pieces2word  # bert词的合并

with open('../AURC/pieces2word.pkl', 'wb') as fw:
    pkl.dump(pieces2word, fw, pkl.HIGHEST_PROTOCOL)