# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import string
#nlp = spacy.load('en_core_web_md')
nlp = spacy.load('en_core_web_sm')


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    print(document.text.split())
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')#构建邻接矩阵。
    
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    return matrix

def process(filename):
    ''' 所有的标点有用空格分开 '''
    print(string.ascii_lowercase)
    print(len(string.ascii_lowercase))
    print(type(string.ascii_lowercase))

    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open('dataset_twitter/trump/test.txt'+'.graph', 'wb')
    fw = open('dataset_twitter/trump/test.txt','w',encoding='utf-8' )
    for i in range(0, len(lines)):
        assert len(lines[i].split('\t'))==4
        text = lines[i].split('\t')[0].lower().strip()
        text_ = ""
        for j in text:
            if j not in (string.ascii_lowercase + string.digits):
                text_ += " " + j + " "
            else:
                text_ += j
        text_=' '.join(text_.split())
        print(text_)
        print("Trump"+"\t"+text_+"\t"+lines[i].split('\t')[1].lower().strip()+"\n")
        #print("Trump"+"\t"+text_+"\t"+lines[i].split('\t')[1].lower().strip()+"\t"+lines[i].split('\t')[2].lower().strip()+"\t"+lines[i].split('\t')[3].lower().strip()+"\n")
        fw.write("Trump"+"\t"+text_+"\t"+lines[i].split('\t')[1].lower().strip()+"\n")
        #fw.write("Trump"+"\t"+text_+"\t"+lines[i].split('\t')[1].lower().strip()+"\t"+lines[i].split('\t')[2].lower().strip()+"\t"+lines[i].split('\t')[3].lower().strip()+"\n")
        adj_matrix = dependency_adj_matrix(text_)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()

def process1(filename):
    '''利用的conditional LSTM的预处理，没有分开@，#等 '''
    pkl_file = open(filename, 'rb')
    lines = pickle.load(pkl_file)
    idx2graph = {}
    fout = open(filename + '.graph', 'wb')
    for i in range(0, len(lines)):
        text = lines[i]
        adj_matrix = dependency_adj_matrix(text)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()

def process2(filename):
    ''' 所有的标点有用空格分开 '''
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open('train.txt'+'.graph', 'wb')
    fw = open('train.txt','w',encoding='utf-8' )
    for i in range(0, len(lines)):
        assert len(lines[i].split('\t'))==6
        text = lines[i].split('\t')[2].lower().strip()

        print(text)
        #文本替换成了小写，可以选择是否为原形。
        fw.write(lines[i].split('\t')[0].lower().strip()+"\t"+lines[i].split('\t')[1].lower().strip()+"\t"+text+"\t"+lines[i].split('\t')[3].lower().strip()+"\t"+lines[i].split('\t')[4].lower().strip()+"\t"+lines[i].split('\t')[5].lower().strip()+"\n")

        adj_matrix = dependency_adj_matrix(text)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()
if __name__ == '__main__':
    process('dataset_twitter/trump_stance_bert_test.txt')
    #process2('./datasets/stance_x/train_yx.txt')
    #process('./datasets/stance/testdata-taskA-all-annotations.txt')
    #process('./datasets/stance/trainingdata-all-annotations.txt')
    #process1('./datasets/stance/test_tweet.pkl')
