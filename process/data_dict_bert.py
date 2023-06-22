#!/usr/bin/env python
import json
from itertools import chain
import pandas as pd
import numpy as np
from transformers import BertTokenizer
# import spacy
# nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('../model_base/')

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
for k,v in data_dicts.items():
    data_dict=v
    sentence = data_dict['sentence'].lower()
    #sentence = data_dict['tokenized_sentence_spacy'].lower()
    idsw = tokenizer.encode(sentence, add_special_tokens=False)
    tokenized_sentence_bert = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=False))

    label = ''
    as_label = ''
    pos = 0
    for t in range(len(sentence)):
        if sentence[t] == ' ':
            label = label + ''
            as_label = as_label + ''
        else:
            label = label + a1[int(k)][3][t]
            as_label = as_label + a1[int(k)][4][t]


    tokenized_sentence_bert_labels = []
    pos = 0
    for t in tokenized_sentence_bert:
        t = t.replace("##", "")
        l = label[pos:pos + len(t)]
        L = l
        assert len(list(set(l))) == 1  # only one stance
        tokenized_sentence_bert_labels.append(L)
        pos += len(t)
    assert len(tokenized_sentence_bert_labels) == len(tokenized_sentence_bert)

    tokenized_sentence_bert_as_labels = []
    pos = 0
    for t in tokenized_sentence_bert:
        t = t.replace("##", "")
        l = as_label[pos:pos + len(t)]
        L = l
        assert len(list(set(l))) == 1  # only one stance#每个词确保只有一个立场极性.
        tokenized_sentence_bert_as_labels.append(L)
        pos += len(t)
    assert len(tokenized_sentence_bert_as_labels) == len(tokenized_sentence_bert)
    data_dict['tokenized_sentence_bert']=' '.join(tokenized_sentence_bert)
    data_dict['tokenized_sentence_bert_labels']=' '.join(tokenized_sentence_bert_labels)
    data_dict['tokenized_sentence_bert_as_labels']=' '.join(tokenized_sentence_bert_as_labels)
    #tokenized_sentence_spacy= data_dict['tokenized_sentence_spacy']
    #tokenized_sentence_spacy_pos= data_dict['tokenized_sentence_spacy_pos']
    # all_words = []
    # for t in tokenized_sentence_spacy.split():
    #     a = tokenizer.convert_ids_to_tokens(tokenizer.encode(t, add_special_tokens=False))
    #     all_words.append(a)
    # # print(all_words)
    # assert len(all_words) == len(tokenized_sentence_spacy.split()) == len(tokenized_sentence_spacy_pos.split())
    # count_find = 0
    # count_find1 = 0
    #
    # tokenized_sentence_bert_pos = []
    # for i in all_words:
    #     for j in i:
    #         assert j == tokenized_sentence_bert[count_find]
    #         tokenized_sentence_bert_pos.append(tokenized_sentence_spacy_pos.split()[count_find1])
    #         count_find += 1
    #     count_find1 += 1
    # print(tokenized_sentence_spacy_pos)
    # print(tokenized_sentence_bert_pos)
    #assert len(tokenized_sentence_bert_pos) == len(tokenized_sentence_bert)


    data_dicts_bert[k]=data_dict

# with open('../data/data_dict.json', 'w',encoding='utf-8') as fp:
#    json.dump(data_dicts, fp)
with open('../data/data_dict_bert.json', 'w') as my_file:
    json.dump(data_dicts, my_file, sort_keys=True, indent=4, separators=(',', ': '))

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
