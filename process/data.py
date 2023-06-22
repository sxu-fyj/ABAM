#!/usr/bin/env python

import pandas as pd
import numpy as np
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
a6 = np.array(ABAM_SEGMENTS)
print(a6[1])
aaa = np.array(ABAM_DATA_SEGMENTS[['topic', 'segment', 'stance', 'inner']]).tolist()
train_datas=[]
dev_datas=[]
test_datas=[]
topic_dict={'abortion':0, 'cloning':1, 'death penalty':2, 'gun control':3, 'marijuana legalization':4, 'minimum wage':5, 'nuclear energy':6, 'school uniforms':7}
stance_dict={'con':0, 'pro':1}
for i in aaa:
    if i[3] == 'Train':
        train_datas.append([topic_dict[i[0]], stance_dict[i[2]], i[1]])
    if i[3] == 'Dev':
        dev_datas.append([topic_dict[i[0]], stance_dict[i[2]], i[1]])
    if i[3] == 'Test':
        test_datas.append([topic_dict[i[0]], stance_dict[i[2]], i[1]])
from pandas.core.frame import DataFrame
train_datas = DataFrame(train_datas, columns=['topic', 'stance', 'sentence'])
train_datas.to_csv('train.csv', index = 0, columns=['topic', 'stance', 'sentence'])
dev_datas = DataFrame(dev_datas, columns=['topic', 'stance', 'sentence'])
dev_datas.to_csv('dev.csv', index = 0, columns=['topic', 'stance', 'sentence'])
test_datas = DataFrame(test_datas, columns=['topic', 'stance', 'sentence'])
test_datas.to_csv('test.csv', index = 0, columns=['topic', 'stance', 'sentence'])
print(ABAM_DATA_SEGMENTS[['topic', 'segment_hash']].groupby('topic').count())
