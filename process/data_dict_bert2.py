#!/usr/bin/env python
import json
from itertools import chain
import pickle
import pandas as pd
import numpy as np
from transformers import BertTokenizer
# import spacy
# nlp = spacy.load("en_core_web_sm")
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
                    matrix[child.i][token.i] = 1
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


data_dicts=json.load(open('../data/data_dict.json','r',encoding='utf-8'))
data_dicts_bert={}
#topic	sentence_hash	sentence	stance	aspect	inner	cross
pieces2word={}
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
    pieces2word[int(k)]=_pieces2word  # bert词的合并

    data_dicts_bert[int(k)]=data_dict
import pickle
with open('../data/pieces2word.pkl', 'wb') as fw:
    pickle.dump(pieces2word, fw, pickle.HIGHEST_PROTOCOL)
# with open('../data/data_dict.json', 'w',encoding='utf-8') as fp:
#    json.dump(data_dicts, fp)
with open('../data/data_dict_bert_pieces2word.json', 'w') as my_file:
    json.dump(data_dicts_bert, my_file, sort_keys=True, indent=4, separators=(',', ': '))
    # json.dump(data_dicts, my_file, sort_keys=True, indent=4, separators=(',', ': '))

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
