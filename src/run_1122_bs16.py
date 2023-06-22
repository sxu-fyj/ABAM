#!/usr/bin/env python
#增加start-transformer，synlstm，stance detection,span2term
#span2term部分，使用句子来抽象到span部分。span外为表示0
#增加字符表示与pos
import os
import pickle as pkl
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import datetime as dt
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
# from sty import fg, bg, ef, rs, RgbFg

# import spacy
# spacy.prefer_gpu()
# nlp = spacy.load("en_core_web_sm")
# from spacy.gold import align

from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)

from transformers import (AdamW, BertConfig, BertTokenizer,
                          get_linear_schedule_with_warmup,
                          BertModel, BertPreTrainedModel)

from utils import InputFeatures_char_word_pos2, get_data_with_labels2

from models import TokenBERT_1122
from models import TokenBERT_LSTM_CRF_start_T



with open('../data/pieces2word.pkl', 'rb') as fr:
    pieces2words = pkl.load(fr)

with open('../data/spacy.txt.graph', 'rb') as fr:
    graph = pkl.load(fr)
    print(1)

def training(train_dataloader, model, device, optimizer, scheduler, max_grad_norm):
    model.train()
    total_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    predictions_train, true_labels_train = [], []
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_input_mask, batch_sentence_ids, batch_sent_length, batch_char_ids, batch_char_len, batch_pos_ids, batch_pieces2word, batch_graph, batch_label_mask, batch_as_mask, batch_am_mask, batch_AM_label_ids, batch_AS_label_ids, batch_topic, batch_token_spacy_ids = batch
        # forward pass



        starts = []
        ends = []
        label_stances = []
        for i in batch_AM_label_ids.cpu().tolist():
            start = []
            end = []
            label_stance = []
            for j in range(len(i)):
                if i[j] == 1 or i[j] == 4:
                    start.append(j)
                    if i[j] == 1:
                        label_stance.append(0)
                    elif i[j] == 4:
                        label_stance.append(1)
                if i[j] == 3 or i[j] == 6:
                    end.append(j)
            starts.append(start)
            ends.append(end)
            label_stances.append(label_stance)



        loss1, logits_AM1, logits_AS1 = model(
            model = 'embedding',
            input_ids=batch_input_ids,
            token_type_ids=batch_sentence_ids,
            attention_mask=batch_input_mask,
            sent_length=batch_sent_length,
            char_ids=batch_char_ids,
            char_len=batch_char_len,
            pos_ids=batch_pos_ids,
            pieces2word=batch_pieces2word,
            graph=batch_graph,
            label_mask=batch_label_mask,
            AM_labels=batch_AM_label_ids,
            AS_labels=batch_AS_label_ids
        )

        loss2, stance_softmax = model(
            model='stance',
            input_ids=batch_input_ids,
            token_type_ids=batch_sentence_ids,
            attention_mask=batch_input_mask,
            sent_length=batch_sent_length,
            char_ids=batch_char_ids,
            char_len=batch_char_len,
            pos_ids=batch_pos_ids,
            pieces2word=batch_pieces2word,
            graph=batch_graph,
            label_mask=batch_label_mask,
            AM_labels=batch_AM_label_ids,
            AS_labels=batch_AS_label_ids,
            starts=starts,
            ends=ends,
            label_stance=label_stances,
            topic_indices=batch_topic,
            token_spacy_ids=batch_token_spacy_ids
        )
        stance_softmax = stance_softmax.cpu().tolist()
        logits_AM2 = np.zeros((batch_AM_label_ids.shape[0], batch_AM_label_ids.shape[1], 7))
        span_count = 0
        for i in range(len(starts)):
            for j in range(len(starts[i])):
                for k in range(batch_AM_label_ids.shape[1]):
                    if k == starts[i][j]:
                        if stance_softmax[span_count][0]>stance_softmax[span_count][1]:
                            logits_AM2[i][starts[i][j]][1]=stance_softmax[span_count][0]
                        else:
                            logits_AM2[i][starts[i][j]][4]=stance_softmax[span_count][1]
                    if k == ends[i][j]:
                        if stance_softmax[span_count][0] > stance_softmax[span_count][1]:
                            logits_AM2[i][ends[i][j]][3]=stance_softmax[span_count][0]
                        else:
                            logits_AM2[i][ends[i][j]][6]=stance_softmax[span_count][1]
                    if starts[i][j] < k < ends[i][j]:
                        if stance_softmax[span_count][0] > stance_softmax[span_count][1]:
                            logits_AM2[i][k][2]=stance_softmax[span_count][0]
                        else:
                            logits_AM2[i][k][5]=stance_softmax[span_count][1]
                    if starts[i][j] > k or k > ends[i][j]:
                        logits_AM2[i][k][0]=1.0

                span_count += 1
                # print(logits_AM2[i])
                # print(span_count)



                # logits_AM2[i][starts[i]]=stance_softmax[i][j]
        loss3, logits_AS2 = model(
            model='span2term',
            input_ids=batch_input_ids,
            token_type_ids=batch_sentence_ids,
            attention_mask=batch_input_mask,
            sent_length=batch_sent_length,
            char_ids=batch_char_ids,
            char_len=batch_char_len,
            pos_ids=batch_pos_ids,
            pieces2word=batch_pieces2word,
            graph=batch_graph,
            label_mask=batch_label_mask,
            AM_labels=batch_AM_label_ids,
            AS_labels=batch_AS_label_ids,
            span2term=batch_am_mask,
        )

        loss4 = model(
            model='final',
            input_ids=batch_input_ids,
            token_type_ids=batch_sentence_ids,
            attention_mask=batch_input_mask,
            sent_length=batch_sent_length,
            char_ids=batch_char_ids,
            char_len=batch_char_len,
            pos_ids=batch_pos_ids,
            pieces2word=batch_pieces2word,
            graph=batch_graph,
            label_mask=batch_label_mask,
            AM_labels=batch_AM_label_ids,
            AS_labels=batch_AS_label_ids,
            logits_AM1 = logits_AM1,
            logits_AM2 = logits_AM2,
            logits_AS1 = logits_AS1,
            logits_AS2 = logits_AS2
        )

        loss = loss1 + loss2 + loss3 + loss4
        #print(loss)
        # backward pass
        loss.backward()

        # track train loss
        total_loss += loss.item()
        nb_tr_examples += batch_input_ids.size(0)
        nb_tr_steps += 1

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # update learning rate
        scheduler.step()

        model.zero_grad()

    return model, optimizer, scheduler, total_loss


def evaluation(sample_dataloader, model, device, tokenizer):
    model.eval()
    max_spacy_length = 85
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []

    y_output_true = []
    y_output_pred = []

    for step, batch in enumerate(sample_dataloader):
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_input_mask, batch_sentence_ids, batch_sent_length, batch_char_ids, batch_char_len, batch_pos_ids, batch_pieces2word, batch_graph, batch_label_mask, batch_as_mask, batch_am_mask, batch_AM_label_ids, batch_AS_label_ids, batch_topic, batch_token_spacy_ids = batch

        #batch_input_ids, batch_input_mask, batch_sentence_ids, batch_AM_label_ids, batch_AS_label_ids = batch

        with torch.no_grad():
            AM_tags1, AS_tags1, logits_AM1, logits_AS1 = model(
                model = 'embedding',
                input_ids=batch_input_ids,
                token_type_ids=batch_sentence_ids,
                attention_mask=batch_input_mask,
                sent_length = batch_sent_length,
                char_ids=batch_char_ids,
                char_len=batch_char_len,
                pos_ids=batch_pos_ids,
                pieces2word = batch_pieces2word,
                graph=batch_graph,
                label_mask=batch_label_mask
            )

            starts = []
            ends = []
            np_arr = np.array(batch_AM_label_ids.cpu().tolist())
            AM_tags1 = np.int64(np_arr > 0)
            for i in AM_tags1:
                start = []
                end = []
                before = 0
                for j in range(len(i)):
                    if i[j] != before and i[j] != 0:
                        start.append(j)
                    if j != len(i) - 1:
                        if i[j] != i[j + 1] and i[j] != 0:
                            end.append(j)
                    else:
                        if i[j] != 0:
                            end.append(j)
                    before = i[j]
                if len(start) != 0 and len(end) != 0 and len(start) == len(end):
                    starts.append(start)
                    ends.append(end)


            logits_AM2 = np.zeros((len(AM_tags1), max_spacy_length, 7))

            if len(starts) != 0:
                stance_tags2, stance_softmax = model(
                    model='stance',
                    input_ids=batch_input_ids,
                    token_type_ids=batch_sentence_ids,
                    attention_mask=batch_input_mask,
                    sent_length=batch_sent_length,
                    char_ids=batch_char_ids,
                    char_len=batch_char_len,
                    pos_ids=batch_pos_ids,
                    pieces2word=batch_pieces2word,
                    graph=batch_graph,
                    label_mask=batch_label_mask,
                    starts=starts,
                    ends=ends,
                    topic_indices=batch_topic,
                    token_spacy_ids=batch_token_spacy_ids
                )
                stance_softmax = stance_softmax.cpu().tolist()
                span_count = 0
                for i in range(len(starts)):
                    for j in range(len(starts[i])):
                        for k in range(len(AM_tags1[0])):
                            if k == starts[i][j]:
                                if stance_softmax[span_count][0] > stance_softmax[span_count][1]:
                                    logits_AM2[i][starts[i][j]][1] = stance_softmax[span_count][0]
                                else:
                                    logits_AM2[i][starts[i][j]][4] = stance_softmax[span_count][1]
                            if k == ends[i][j]:
                                if stance_softmax[span_count][0] > stance_softmax[span_count][1]:
                                    logits_AM2[i][ends[i][j]][3] = stance_softmax[span_count][0]
                                else:
                                    logits_AM2[i][ends[i][j]][6] = stance_softmax[span_count][1]
                            if starts[i][j] < k < ends[i][j]:
                                if stance_softmax[span_count][0] > stance_softmax[span_count][1]:
                                    logits_AM2[i][k][2] = stance_softmax[span_count][0]
                                else:
                                    logits_AM2[i][k][5] = stance_softmax[span_count][1]
                            if starts[i][j] > k or k > ends[i][j]:
                                logits_AM2[i][k][0] = 1.0
                        span_count += 1


            # batch_am_mask
            AM_tags1s = []
            for i in AM_tags1:
                if len(set(i)) != 1:
                    am_label_mask = [1 if label != 0 else 0 for label in i]
                    AM_tags1s.append(am_label_mask[:max_spacy_length] + [0] * max(0, max_spacy_length - len(am_label_mask)))
                else:
                    AM_tags1s.append(i[:max_spacy_length] + [0] * max(0, max_spacy_length - len(i)))
            AM_tags1s_t = torch.tensor([f for f in AM_tags1s], dtype=torch.float).to(device)
            #print(AM_tags1s)

            AS_tags2, logits_AS2 = model(
                model='span2term',
                input_ids=batch_input_ids,
                token_type_ids=batch_sentence_ids,
                attention_mask=batch_input_mask,
                sent_length=batch_sent_length,
                char_ids=batch_char_ids,
                char_len=batch_char_len,
                pos_ids=batch_pos_ids,
                pieces2word=batch_pieces2word,
                graph=batch_graph,
                label_mask=batch_label_mask,
                span2term=AM_tags1s_t
            )

            batch_AM_tags, batch_AS_tags = model(
                model='final',
                input_ids=batch_input_ids,
                token_type_ids=batch_sentence_ids,
                attention_mask=batch_input_mask,
                sent_length=batch_sent_length,
                char_ids=batch_char_ids,
                char_len=batch_char_len,
                pos_ids=batch_pos_ids,
                pieces2word=batch_pieces2word,
                graph=batch_graph,
                label_mask=batch_label_mask,
                logits_AM1=logits_AM1,
                logits_AM2=logits_AM2,
                logits_AS1=logits_AS1,
                logits_AS2=logits_AS2
            )


        batch_correct_label_ids = []
        batch_correct_tags = []

        output_correct_trues = []
        output_correct_pres = []
        AM_label_dict = {0: 'O', 1: 'B-c', 2: 'I-c', 3: 'E-c', 4: 'B-p', 5: 'I-p', 6: 'E-p'}
        AS_label_dict = {0: 'O', 1: 'B-aspect', 2: 'I-aspect', 3: 'E-aspect'}
        # AMS_label_dict = {'nn': 0, 'na': 0, 'cn': 1, 'ca': 2, 'pn': 3, 'pa': 4}

        for input_ids, AM_label_ids, AS_label_ids, AM_tags, AS_tags in zip(batch_label_mask, batch_AM_label_ids,
                                                                           batch_AS_label_ids, batch_AM_tags,
                                                                           batch_AS_tags):
            input_ids = input_ids.cpu().tolist()
            AM_label_ids = AM_label_ids.cpu().tolist()
            AS_label_ids = AS_label_ids.cpu().tolist()
            if type(AM_tags) != list:
                AM_tags = AM_tags.cpu().tolist()
                AS_tags = AS_tags.cpu().tolist()
            aaa = [AM_label_dict[t] for t in AM_label_ids]
            bbb = [AS_label_dict[t] for t in AS_label_ids]

            aaaa = [AM_label_dict[t] for t in AM_tags]
            bbbb = [AS_label_dict[t] for t in AS_tags]
            seq_len = [i for i, t in enumerate(input_ids) if t == 0][0]  # 102 is the [SEP] token_id
            # input_tokens = tokenizer.convert_ids_to_tokens(input_ids)#
            # correct_label_ids = [AMS_label_dict[l1 + l2] for t, l1, l2 in
            #                      zip(input_tokens[1:seq_len], aaa[1:seq_len], bbb[1:seq_len]) if not t.startswith('##')]
            # correct_tags = [AMS_label_dict[l1 + l2] for t, l1, l2 in
            #                 zip(input_tokens[1:seq_len], aaaa[1:seq_len], bbbb[1:seq_len]) if not t.startswith('##')]

            correct_label_AM = aaa[0:seq_len]
            correct_label_AS = bbb[0:seq_len]
            correct_AM_tags = aaaa[0:seq_len]
            correct_AS_tags = bbbb[0:seq_len]
            batch_correct_label_ids.append(correct_label_AM)
            batch_correct_label_ids.append(correct_label_AS)
            batch_correct_tags.append(correct_AM_tags)
            batch_correct_tags.append(correct_AS_tags)

            correct_label_ids = [l1 + l2 for l1, l2 in zip(aaa[0:seq_len], bbb[0:seq_len])]
            correct_tags = [l1 + l2 for l1, l2 in zip(aaaa[0:seq_len], bbbb[0:seq_len])]

            output_correct_trues.append(correct_label_ids)
            output_correct_pres.append(correct_tags)

            assert len(batch_correct_label_ids) == len(batch_correct_tags)
        y_true += batch_correct_label_ids
        y_pred += batch_correct_tags

        y_output_true += output_correct_trues
        y_output_pred += output_correct_pres
    # flatten
    # YT, YP = [], []
    # for t, p in zip(y_true, y_pred):
    #     YT += t
    #     YP += p
    # assert len(YT) == len(YP)
    #
    #
    #
    # p, r, f1s, s = metrics.precision_recall_fscore_support(y_pred=YP, y_true=YT, average='macro', warn_for=tuple())
    # cm = confusion_matrix(y_pred=YP, y_true=YT)
    # print(cm)

    print(classification_report(y_true, y_pred, digits=4))
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return y_output_true, y_output_pred, micro_f1, macro_f1


def main():
    parser = argparse.ArgumentParser(description='Run Fine-Grained Argument Unit Recognition and Classification.')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    parser.add_argument('--card_number', type=int, default=0, help='Your GPU card number.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of labels; either 3 (pro, con, non) or 2 (arg, non).')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='The learning rate.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='The max_grad_norm.')
    parser.add_argument('--max_sequence_length', type=int, default=110,
                        help='The maximum sequence length of the input text.')#71,91
    parser.add_argument('--max_spacy_length', type=int, default=85,
                        help='The maximum sequence length of the input text.')  # spacy分词的长度
    parser.add_argument('--train_batch_size', type=int, default=16, help='Your train set batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Your evaluation set batch size.')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Your test set batch size.')
    parser.add_argument('--target_domain', type=str, default='inner', help='inner OR cross')
    parser.add_argument('--input_file', type=str, default='../data/data_dict_bert_pieces2word.json', help='The input dict file.')
    parser.add_argument('--data_dir', type=str, default='../data/', help='The data directory.')
    parser.add_argument('--output_dir', type=str, default='../models_final1122_bs16_notopic_2/',
                        help='The output directory of the model, config and predictions.')
    parser.add_argument('--pretrained_weights', type=str, default='bert-large-cased-whole-word-masking',
                        help='The pretrained bert model.')
    parser.add_argument("--fine_tuning", default=True, action="store_true", help="Flag for full fine-tuning.")
    parser.add_argument("--crf", default=True, action="store_true", help="Flag for CRF useage.")
    parser.add_argument("--train", default=True, action="store_true", help="Flag for training.")
    parser.add_argument("--eval", default=True, action="store_true", help="Flag for evaluation.")
    parser.add_argument("--save_model", default=True, action="store_true", help="Flag for saving.")
    parser.add_argument("--save_prediction", default=True, action="store_true", help="Flag for saving.")
    args = parser.parse_args()

    #############################################################################
    # Parameters and paths
    device = torch.device("cuda:{}".format(args.card_number) if torch.cuda.is_available() else "cpu")
    print("device", device)

    task = '_'.join(['aurc', args.target_domain[:2].lower()])

    MODEL_PATH = os.path.join(args.output_dir, '{}_token.pt'.format(task))
    print(MODEL_PATH)
    CONFIG_PATH = os.path.join(args.output_dir, '{}_config.json'.format(task))
    print(CONFIG_PATH)
    PREDICTIONS_DEV = os.path.join(args.output_dir, '{}_predictions_dev.json'.format(task))
    print(PREDICTIONS_DEV)
    PREDICTIONS_TEST = os.path.join(args.output_dir, '{}_predictions_test.json'.format(task))
    print(PREDICTIONS_TEST)

    #############################################################################
    # Load Data
    fname = os.path.join(args.data_dir, args.input_file)
    print(fname)
    with open(fname, 'r') as my_file:
        AURC_DATA_dict = json.load(my_file)
    print('数据集大小')
    print(len(AURC_DATA_dict))
    topic = []
    for k, v in AURC_DATA_dict.items():
        topic.append(v['topic'])
    topics = sorted(set(topic))
    topics_dict = {}
    for i in range(len(topics)):
        topics_dict[topics[i]] = i
    print(len(topics), topics)
    from collections import Counter
    c = Counter(topic)
    print(dict(c))
    label2id = dict()
    label2id['O'] = 0
    label2id['B-c'] = 1
    label2id['I-c'] = 2
    label2id['E-c'] = 3
    label2id['B-p'] = 4
    label2id['I-p'] = 5
    label2id['E-p'] = 6

    aslabel2id = dict()
    aslabel2id['O'] = 0
    aslabel2id['B-aspect'] = 1
    aslabel2id['I-aspect'] = 2
    aslabel2id['E-aspect'] = 3

#词性

    tokenizer = BertTokenizer.from_pretrained('../model_base/')

    train_features = []
    eval_features = []
    test_features = []

    all_input_tokens_dev = []
    all_input_tokens_test = []
    max_tok=0
    char=[]

    max_char_len = 20

    char_dict = {'I': 0, 'n': 1, 'o': 2, 'r': 3, 'd': 4, 'e': 5, 't': 6, 'i': 7, 's': 8, 'u': 9, 'a': 10, 'w': 11, 'm': 12, 'f': 13,
     'b': 14, ',': 15, '-': 16, 'c': 17, 'h': 18, 'l': 19, 'v': 20, 'k': 21, 'p': 22, 'y': 23, '.': 24, 'C': 25,
     '"': 26, 'B': 27, ':': 28, "'": 29, 'g': 30, 'L': 31, 'x': 32, ';': 33, '5': 34, '%': 35, 'Y': 36, 'j': 37,
     'A': 38, 'R': 39, '1': 40, '9': 41, 'T': 42, '8': 43, '0': 44, 'z': 45, 'S': 46, 'q': 47, 'J': 48, 'G': 49,
     '7': 50, '6': 51, 'U': 52, 'P': 53, 'M': 54, '2': 55, '4': 56, 'W': 57, 'D': 58, 'N': 59, 'H': 60, '(': 61,
     ')': 62, 'E': 63, '[': 64, ']': 65, '?': 66, '&': 67, 'K': 68, 'F': 69, '≈': 70, 'O': 71, '!': 72, '3': 73,
     '/': 74, 'V': 75, '$': 76, '*': 77, 'Q': 78, 'Z': 79, '<': 80, '=': 81, '>': 82, '−': 83, '#': 84, '+': 85,
     '~': 86, '_': 87, 'X': 88, 'ā': 89, 'à': 90, '«': 91, '»': 92, '£': 93, 'padding': 94}

    POS_dict = {'ADP': 0, 'NOUN': 1, 'PART': 2, 'VERB': 3, 'PUNCT': 4, 'ADJ': 5, 'PRON': 6, 'CCONJ': 7, 'ADV': 8, 'SCONJ': 9, 'DET': 10, 'AUX': 11, 'PROPN': 12, 'NUM': 13, 'INTJ': 14, 'SYM': 15, 'X': 16, 'pad':17}
    # {'ADP': 10550, 'NOUN': 27584, 'PART': 3318, 'VERB': 15132, 'PUNCT': 11156, 'ADJ': 11281, 'PRON': 3660, 'CCONJ': 3602, 'ADV': 5306, 'SCONJ': 3769, 'DET': 11479, 'AUX': 5846, 'PROPN': 3118, 'NUM': 1053, 'INTJ': 22, 'SYM': 126, 'X': 47}
    all_tokens={}
    all_tokens["padding"]=0
    tokens=[]
    for count, AD in AURC_DATA_dict.items():
        a = AD['tokenized_sentence_spacy'].split()
        tokens.extend(a)
    tokens = sorted(set(tokens))

    ccc = Counter(tokens)
    count_token=1
    for k in dict(ccc):
        all_tokens[k]=count_token
        count_token+=1
    print(len(all_tokens))
    print(count_token)
    for count, AD in AURC_DATA_dict.items():
        sequence_dict = tokenizer.encode_plus(AD['tokenized_sentence_spacy'], max_length=args.max_sequence_length,
                                              pad_to_max_length=True, add_special_tokens=True)
        sequence_dict['batch_topic']=topics_dict[AD['topic']]


        aaa = tokenizer.convert_ids_to_tokens(sequence_dict['input_ids'])
        # sequence_dict_ss = tokenizer.encode_plus(AD['tokenized_sentence_spacy'], max_length=args.max_sequence_length,
        #                                           pad_to_max_length=True, add_special_tokens=True)#
        # cccc = tokenizer.convert_ids_to_tokens(sequence_dict_ss['input_ids'])
        tokens = [tokenizer.tokenize(word) for word in AD['tokenized_sentence_spacy'].split()]#
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = [tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id]
        bert_len=len(_bert_inputs)
        _bert_inputs2 = _bert_inputs[:args.max_sequence_length] + [0] * max(0, args.max_sequence_length - len(_bert_inputs))

        #argument unit label, BIEO
        IO_AM_label = ["I-"+label[0] if label[0]!="n" else label[0] for label in AD['tokenized_sentence_spacy_labels'].split(' ')]
        BIEO_AM_label = []
        for am in range(len(IO_AM_label)):
            if "I-" in IO_AM_label[am]:
                if am==0:
                    BIEO_AM_label.append(IO_AM_label[am].replace('I-','B-'))
                    continue
                elif am==len(IO_AM_label)-1:
                    BIEO_AM_label.append(IO_AM_label[am].replace('I-','E-'))
                    continue
                else:
                    if IO_AM_label[am] != IO_AM_label[am-1]:
                        BIEO_AM_label.append(IO_AM_label[am].replace('I-', 'B-'))
                        continue
                    elif IO_AM_label[am] != IO_AM_label[am+1]:
                        BIEO_AM_label.append(IO_AM_label[am].replace('I-', 'E-'))
                        continue
                    else:
                        BIEO_AM_label.append(IO_AM_label[am])
                        continue
            else:
                BIEO_AM_label.append('O')
                continue
        assert len(BIEO_AM_label) == len(IO_AM_label)

        input_am_labels = [label2id[label] for label in BIEO_AM_label]
        am_label_len = len(input_am_labels)
        input_am_labels = input_am_labels[:args.max_spacy_length] + [0] * max(0, args.max_spacy_length - len(input_am_labels))
        sequence_dict['AM_label_ids'] = input_am_labels


        # input_am_labels = [label2id[label[0]] for label in AD['tokenized_sentence_spacy_labels'].split(' ')]
        # am_label_len=len(input_am_labels)
        # input_am_labels = input_am_labels[:args.max_spacy_length] + [0] * max(0, args.max_spacy_length - len(input_am_labels))
        # sequence_dict['AM_label_ids'] = input_am_labels


        # argument unit aspect term label
        IO_AS_label = ["I-aspect" if label[0] != "0" else "O" for label in AD['tokenized_sentence_spacy_as_labels'].split(' ')]
        BIEO_AS_label = []
        for am in range(len(IO_AS_label)):
            if "I-" in IO_AS_label[am]:
                if am == 0:
                    BIEO_AS_label.append(IO_AS_label[am].replace('I-', 'B-'))
                    continue
                elif am == len(IO_AM_label) - 1:
                    BIEO_AS_label.append(IO_AS_label[am].replace('I-', 'E-'))
                    continue
                else:
                    if IO_AS_label[am] != IO_AS_label[am - 1]:
                        BIEO_AS_label.append(IO_AS_label[am].replace('I-', 'B-'))
                        continue
                    elif IO_AS_label[am] != IO_AS_label[am + 1]:
                        BIEO_AS_label.append(IO_AS_label[am].replace('I-', 'E-'))
                        continue
                    else:
                        BIEO_AS_label.append(IO_AS_label[am])
                        continue
            else:
                BIEO_AS_label.append('O')
                continue
        assert len(BIEO_AS_label) == len(IO_AS_label)

        input_as_labels = [aslabel2id[label] for label in BIEO_AS_label]
        as_label_len = len(input_as_labels)
        input_as_labels = input_as_labels[:args.max_spacy_length] + [0] * max(0, args.max_spacy_length - len(input_as_labels))
        sequence_dict['AS_label_ids'] = input_as_labels  # 方面项的标签


        # input_as_labels = [int(label[0]) for label in AD['tokenized_sentence_spacy_as_labels'].split(' ')]
        # as_label_len=len(input_as_labels)
        # input_as_labels = input_as_labels[:args.max_spacy_length] + [0] * max(0, args.max_spacy_length - len(input_as_labels))
        # sequence_dict['AS_label_ids'] = input_as_labels#方面项的标签

        label_mask = [1 for label in AD['tokenized_sentence_spacy_as_labels'].split(' ')]
        label_mask_len =len(label_mask)
        label_mask = label_mask[:args.max_spacy_length] + [0] * max(0, args.max_spacy_length - len(label_mask))
        sequence_dict['label_mask'] = label_mask  # 方面项的标签

        if "1" not in AD['tokenized_sentence_spacy_as_labels']:
            as_label_mask = [0 if label[0] != "0" else 0 for label in AD['tokenized_sentence_spacy_as_labels'].split(' ')]
        else:
            as_label_mask = [1 if label[0] != "0" else 0 for label in AD['tokenized_sentence_spacy_as_labels'].split(' ')]
        as_label_mask = as_label_mask[:args.max_spacy_length] + [0] * max(0, args.max_spacy_length - len(as_label_mask))
        sequence_dict['as_label_mask'] = as_label_mask  # 方面项的标签

        am_label_mask = [1 if label[0] != "n" else 0 for label in AD['tokenized_sentence_spacy_labels'].split(' ')]
        am_label_mask = am_label_mask[:args.max_spacy_length] + [0] * max(0, args.max_spacy_length - len(am_label_mask))
        sequence_dict['am_label_mask'] = am_label_mask  # 方面项的标签


        sent_length=len(AD['tokenized_sentence_spacy_as_labels'].split(' '))
        sequence_dict['sent_length'] = sent_length

        sequence_dict['input_tokens'] = tokenizer.convert_ids_to_tokens(_bert_inputs2)
        pieces2word_ = pieces2words[int(count)]

        token_spacy_ids = []
        for i in AD['tokenized_sentence_spacy'].split():
            token_spacy_ids.append(all_tokens[i])
        assert len(token_spacy_ids) == pieces2word_.shape[0]

        token_spacy_ids = token_spacy_ids[:args.max_spacy_length] + [0] * max(0, args.max_spacy_length - len(token_spacy_ids))
        sequence_dict['token_spacy_ids'] = token_spacy_ids

        assert am_label_len == as_label_len == label_mask_len == pieces2word_.shape[0]
        assert len(input_am_labels)==len(input_as_labels)==args.max_spacy_length
        assert bert_len==pieces2word_.shape[1]

        def fill(data, new_data):
            new_data[:data.shape[0], :data.shape[1]] = data
            return new_data

        sub_mat = np.zeros((args.max_spacy_length, args.max_sequence_length), dtype=np.bool)
        pieces2word_ = fill(pieces2word_, sub_mat)




        graph_ = graph[int(count)]

        assert am_label_len == as_label_len == label_mask_len == graph_.shape[0]
        assert len(input_am_labels) == len(input_as_labels) == args.max_spacy_length

        def fill(data, new_data):
            new_data[:data.shape[0], :data.shape[1]] = data
            return new_data

        sub_mat = np.zeros((args.max_spacy_length, args.max_spacy_length), dtype=np.float)
        graph_ = fill(graph_, sub_mat)


        # char
        char = [[char_dict[l] for l in label] for label in AD['tokenized_sentence_spacy'].split(' ')]
        char_padding = [schar[:max_char_len] + [94] * max(0, max_char_len - len(schar)) for schar in char]
        char_padding = char_padding[:args.max_spacy_length] + [[94] * max_char_len] * max(0,  args.max_spacy_length - len(char_padding))
        sequence_dict['char_ids'] = char_padding

        # char len
        char_len = [len(label) for label in AD['tokenized_sentence_spacy'].split(' ')]
        char_len_padding = char_len[:args.max_spacy_length] + [1] * max(0, args.max_spacy_length - len(char_len))
        sequence_dict['char_len'] = char_len_padding


        # pos
        pos_labels = [POS_dict[label] for label in AD['tokenized_sentence_spacy_pos'].split(' ')]
        pos_labels = pos_labels[:args.max_spacy_length] + [17] * max(0, args.max_spacy_length - len(pos_labels))
        sequence_dict['pos_labels'] = pos_labels



        #print(pos_labels)
        #


        # for k, v in sequence_dict.items():
        #     assert len(v) == args.max_sequence_length

        FE = [
            InputFeatures_char_word_pos2(
                input_ids = sequence_dict['input_ids'],
                attention_mask = sequence_dict['attention_mask'],
                token_type_ids = sequence_dict['token_type_ids'],
                word_length = sequence_dict['sent_length'],
                char_ids = sequence_dict['char_ids'],
                char_len = sequence_dict['char_len'],
                pos_ids = sequence_dict['pos_labels'],
                pieces2word=pieces2word_,
                graph = graph_,
                label_mask=sequence_dict['label_mask'],
                as_mask =sequence_dict['as_label_mask'],
                am_mask=sequence_dict['am_label_mask'],
                AM_label_ids=sequence_dict['AM_label_ids'],
                AS_label_ids=sequence_dict['AS_label_ids'],
                topic=sequence_dict['batch_topic'],
                token_spacy_ids = sequence_dict['token_spacy_ids'],
                sentence_hash=AD['sentence_hash']
            )
        ]
        if AD[args.target_domain] == 'Train':
            train_features += FE
        if AD[args.target_domain] == 'Dev':
            eval_features += FE
            all_input_tokens_dev.append(AD['tokenized_sentence_spacy'].split(' '))
        if AD[args.target_domain] == 'Test':
            test_features += FE
            all_input_tokens_test.append(AD['tokenized_sentence_spacy'].split(' '))

    # POS_dict = Counter(POS)
    # print(dict(POS_dict))

    # char_dict = Counter(char)
    # print(dict(char_dict))
    # char_dict2={}
    # count_char=0
    # for k, v in char_dict.items():
    #     char_dict2[k]=count_char
    #     count_char+=1
    # print(char_dict2)
    print(max_tok)
    # TRAIN DATA
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    all_sent_length = torch.tensor([f.word_length for f in train_features], dtype=torch.long)
    all_char_ids = torch.tensor([f.char_ids for f in train_features], dtype=torch.long)
    all_char_len = torch.tensor([f.char_len for f in train_features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in train_features], dtype=torch.long)
    all_pieces2word = torch.tensor([f.pieces2word for f in train_features], dtype=torch.long)
    all_graph = torch.tensor([f.graph for f in train_features], dtype=torch.float)
    all_label_mask = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    all_as_mask = torch.tensor([f.as_mask for f in train_features], dtype=torch.float)
    all_am_mask = torch.tensor([f.am_mask for f in train_features], dtype=torch.float)
    all_AM_label_ids = torch.tensor([f.AM_label_ids for f in train_features], dtype=torch.long)
    all_AS_label_ids = torch.tensor([f.AS_label_ids for f in train_features], dtype=torch.long)
    all_topic = torch.tensor([f.topic for f in train_features], dtype=torch.long)
    all_token_spacy_ids = torch.tensor([f.token_spacy_ids for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sent_length, all_char_ids, all_char_len, all_pos_ids, all_pieces2word, all_graph, all_label_mask, all_as_mask, all_am_mask, all_AM_label_ids, all_AS_label_ids, all_topic, all_token_spacy_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    print(len(train_sampler), len(train_dataloader))

    # EVAL DATA
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
    all_sent_length = torch.tensor([f.word_length for f in eval_features], dtype=torch.long)
    all_char_ids = torch.tensor([f.char_ids for f in eval_features], dtype=torch.long)
    all_char_len = torch.tensor([f.char_len for f in eval_features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in eval_features], dtype=torch.long)
    all_pieces2word = torch.tensor([f.pieces2word for f in eval_features], dtype=torch.long)
    all_graph = torch.tensor([f.graph for f in eval_features], dtype=torch.float)
    all_label_mask = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    all_as_mask = torch.tensor([f.as_mask for f in eval_features], dtype=torch.float)
    all_am_mask = torch.tensor([f.am_mask for f in eval_features], dtype=torch.float)
    all_AM_label_ids = torch.tensor([f.AM_label_ids for f in eval_features], dtype=torch.long)
    all_AS_label_ids = torch.tensor([f.AS_label_ids for f in eval_features], dtype=torch.long)
    all_topic = torch.tensor([f.topic for f in eval_features], dtype=torch.long)
    all_token_spacy_ids = torch.tensor([f.token_spacy_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sent_length, all_char_ids, all_char_len, all_pos_ids, all_pieces2word, all_graph, all_label_mask, all_as_mask, all_am_mask, all_AM_label_ids, all_AS_label_ids, all_topic, all_token_spacy_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    print(len(eval_sampler), len(eval_dataloader))

    # TEST DATA
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long)
    all_sent_length = torch.tensor([f.word_length for f in test_features], dtype=torch.long)
    all_char_ids = torch.tensor([f.char_ids for f in test_features], dtype=torch.long)
    all_char_len = torch.tensor([f.char_len for f in test_features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in test_features], dtype=torch.long)
    all_pieces2word = torch.tensor([f.pieces2word for f in test_features], dtype=torch.long)
    all_graph = torch.tensor([f.graph for f in test_features], dtype=torch.float)
    all_label_mask = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
    all_as_mask = torch.tensor([f.as_mask for f in test_features], dtype=torch.float)
    all_am_mask = torch.tensor([f.am_mask for f in test_features], dtype=torch.float)
    all_AM_label_ids = torch.tensor([f.AM_label_ids for f in test_features], dtype=torch.long)
    all_AS_label_ids = torch.tensor([f.AS_label_ids for f in test_features], dtype=torch.long)
    all_topic = torch.tensor([f.topic for f in test_features], dtype=torch.long)
    all_token_spacy_ids = torch.tensor([f.token_spacy_ids for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_sent_length, all_char_ids, all_char_len, all_pos_ids, all_pieces2word, all_graph, all_label_mask, all_as_mask, all_am_mask, all_AM_label_ids, all_AS_label_ids, all_topic, all_token_spacy_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)
    print(len(test_sampler), len(test_dataloader))

    #############################################################################
    # Training
    if args.train:

        # Load Config
        config = BertConfig.from_pretrained('../model_base/', num_labels=args.num_labels)
        config.hidden_dropout_prob = 0.1

        # Model
        model = TokenBERT_1122(
            model_name='../model_base/',
            num_labels=args.num_labels,
            output_hidden_states=False,
            use_crf=args.crf,
            device=device
        )
        model.to(device)

        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.epochs
        print(num_train_optimization_steps)

        if args.fine_tuning:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(model.tokenbert.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=num_train_optimization_steps)

        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

        # Main Loop
        print("##### DOMAIN:", args.target_domain, ",", "use CRF:", args.crf, ",", "learning-rate:", args.learning_rate,
              ",", "DROPOUT:", config.hidden_dropout_prob)
        print()

        best_p_dev, best_r_dev, best_f1s_dev = 0.0, 0.0, 0.0
        best_epoch = None
        prev_tr_loss = 1e8

        best_y_true_dev, best_y_pred_dev = [], []

        for epoch in range(args.epochs):
            print("Epoch: %4i" % epoch, dt.datetime.now())

            # TRAINING
            model, optimizer, scheduler, tr_loss = training(train_dataloader, model=model, device=device,
                                                            optimizer=optimizer, scheduler=scheduler,
                                                            max_grad_norm=args.max_grad_norm)

            # EVALUATION: TRAIN SET
            y_true_train, y_pred_train, micro_train, macro_train = evaluation(
                train_dataloader, model=model, device=device, tokenizer=tokenizer)
            print("TRAIN:  micro_F1. %.4f |  macro_F1 %.4f" % (micro_train, macro_train))

            # EVALUATION: DEV SET
            y_true_dev, y_pred_dev, micro_dev, macro_dev = evaluation(
                eval_dataloader, model=model, device=device, tokenizer=tokenizer)
            print("EVAL:   micro_F1. %.4f |  macro_F1 %.4f | BEST F1: %.4f" % (micro_dev, macro_dev, best_f1s_dev),
                  best_epoch)

            if micro_dev > best_f1s_dev:
                best_f1s_dev = micro_dev
                best_epoch = epoch

                # EVALUATION: TEST SET
                y_true_test, y_pred_test, micro_test, macro_test = evaluation(
                    test_dataloader, model=model, device=device, tokenizer=tokenizer)
                print("TEST:   micro_F1. %.4f |  macro_F1 %.4f" % (micro_test, macro_test))

                if args.save_model:
                    # Save Model
                    torch.save(model.state_dict(), MODEL_PATH)
                    # Save Config
                    with open(CONFIG_PATH, 'w') as f:
                        json.dump(config.to_json_string(), f, sort_keys=True, indent=4, separators=(',', ': '))

                if args.save_prediction:
                    # SAVE PREDICTED DATA
                    # DEV
                    DATA_DEV = get_data_with_labels2(all_input_tokens_dev, y_true_dev, y_pred_dev)
                    with open(PREDICTIONS_DEV, 'w') as f:
                        json.dump(DATA_DEV, f, sort_keys=True, indent=4, separators=(',', ': '))
                    # TEST
                    DATA_TEST = get_data_with_labels2(all_input_tokens_test, y_true_test, y_pred_test)
                    with open(PREDICTIONS_TEST, 'w') as f:
                        json.dump(DATA_TEST, f, sort_keys=True, indent=4, separators=(',', ': '))
            print()

    #################################################################################
    # Load the fine-tuned model:
    if args.eval:
        # Model
        model = TokenBERT_1122(
            model_name='../model_base/',
            num_labels=args.num_labels,
            output_hidden_states=False,
            use_crf=args.crf,
            device=device
        )
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)

        print("\nInference:\n")

        # EVALUATION: TRAIN SET
        y_true_train, y_pred_train, micro_train, macro_train = evaluation(
            train_dataloader, model=model, device=device, tokenizer=tokenizer)
        print("TRAIN:  micro_F1. %.4f |  macro_F1 %.4f" % (micro_train, macro_train))

        # EVALUATION: DEV SET
        y_true_dev, y_pred_dev, micro_dev, macro_dev = evaluation(
            eval_dataloader, model=model, device=device, tokenizer=tokenizer)
        print("EVAL:   micro_F1. %.4f |  macro_F1 %.4f" % (micro_dev, macro_dev))

        # EVALUATION: TEST SET
        y_true_test, y_pred_test, micro_test, macro_test = evaluation(
            test_dataloader, model=model, device=device, tokenizer=tokenizer)
        print("TEST:   micro_F1. %.4f |  macro_F1 %.4f" % (micro_test, macro_test))


if __name__ == "__main__":
    main()

