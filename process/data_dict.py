#!/usr/bin/env python
from itertools import chain
import pandas as pd
import json
import numpy as np
#from transformers import BertTokenizer
import spacy
nlp = spacy.load("en_core_web_sm")
#tokenizer = BertTokenizer.from_pretrained('../model_base/')

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
#topic	sentence_hash	sentence	stance	aspect	inner	cross
data_dicts={}
count=0
for i in a1:
    data_dict={}
    data_dict['topic']=i[0]
    data_dict['sentence_hash']=i[1]
    data_dict['sentence']=hash_dict[i[1]]
    print(len(hash_dict[i[1]]))
    data_dict['stance']=i[3]
    print(len(i[3]))
    data_dict['aspect']=i[4]
    print(len(i[4]))
    assert len(hash_dict[i[1]])==len(i[3])==len(i[4])
    data_dict['inner']=i[5]
    data_dict['cross']=i[6]
    sentence = hash_dict[i[1]]
    label=''
    as_label=''
    pos = 0
    for t in range(len(sentence)):
        if sentence[t]==' ':
            label = label + ''
            as_label = as_label + ''
        else:
            label = label + i[3][t]
            as_label = as_label + i[4][t]

    print(label)
    print(sentence)
    doc = nlp(sentence)
    a=doc.sents
    tokenized_sentence_spacy = [[token.text for token in s] for s in doc.sents]
    tokenized_sentence_spacy_pos = [[token.pos_ for token in s] for s in doc.sents]
    # tokenized_sentence_spacy_ent_type = [[token.ent_type_ for token in s] for s in doc.sents]
    tokenized_sentence_spacy = list(chain.from_iterable(tokenized_sentence_spacy))
    tokenized_sentence_spacy_pos = list(chain.from_iterable(tokenized_sentence_spacy_pos))
    #assert len(tokenized_sentence_spacy)==1
    tokenized_sentence_spacy_labels = []
    pos = 0
    for t in tokenized_sentence_spacy:
        l = label[pos:pos + len(t)]
        L = l
        assert len(list(set(l))) == 1  # only one stance#每个词确保只有一个立场极性.
        tokenized_sentence_spacy_labels.append(L)
        pos += len(t)

    tokenized_sentence_spacy_as_labels = []
    pos = 0
    for t in tokenized_sentence_spacy:
        l = as_label[pos:pos + len(t)]
        L = l
        assert len(list(set(l))) == 1  # only one stance#每个词确保只有一个立场极性.
        tokenized_sentence_spacy_as_labels.append(L)
        pos += len(t)
    assert len(tokenized_sentence_spacy_labels) == len(tokenized_sentence_spacy) == len(tokenized_sentence_spacy_pos)
    data_dict['tokenized_sentence_spacy_labels']=' '.join(tokenized_sentence_spacy_labels)
    data_dict['tokenized_sentence_spacy_as_labels']=' '.join(tokenized_sentence_spacy_as_labels)
    data_dict['tokenized_sentence_spacy']=' '.join(tokenized_sentence_spacy)
    data_dict['tokenized_sentence_spacy_pos']=' '.join(tokenized_sentence_spacy_pos)

    #tokenized_sentence_spacy = list(chain.from_iterable(tokenized_sentence_spacy))  # flatten nested list if applicable
    #tokenized_sentence_bert = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=False))

    # for t in range(len(sentence)):
    #
    #     l = i[3][pos:pos + len(t)]
    #     L = l
    #     print(t)
    #     print(L)
    #     assert len(list(set(l))) == 1  # only one stance#每个词确保只有一个立场极性.
    #     tokenized_sentence_spacy_labels.append(L)
    #     pos += len(t)
    # assert len(tokenized_sentence_spacy_labels) == len(sentence)

    data_dicts[count]=data_dict
    count+=1

# with open('../data/data_dict.json', 'w',encoding='utf-8') as fp:
#    json.dump(data_dicts, fp)
with open('../data/data_dict.json', 'w') as my_file:
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
