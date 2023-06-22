#!/usr/bin/env python

import torch
from torch import nn

from transformers import BertForTokenClassification
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from transformers import BertModel

class TokenBERT(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True):
        super(TokenBERT, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.fc_linear1 = torch.nn.Linear(768, 3)
        self.fc_linear2 = torch.nn.Linear(768, 2)

        if self.use_crf:
            self.crf1 = CRF(3, batch_first=self.batch_first)
            self.crf2 = CRF(2, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids, AM_labels=None, AS_labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)  # (B, L, H)
        logits_AM = self.fc_linear1(sequence_output)
        logits_AS = self.fc_linear2(sequence_output)

        if self.use_crf:
            if AM_labels is not None: # training
                return -self.crf1(logits_AM, AM_labels, attention_mask.byte())-self.crf2(logits_AS, AS_labels, attention_mask.byte())
            else: # inference
                return self.crf1.decode(logits_AM, attention_mask.byte()), self.crf2.decode(logits_AS, attention_mask.byte())
        # else:
        #     if AM_labels is not None: # training
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(
        #             logits.view(-1, self.num_labels),
        #             labels.view(-1)
        #         )
        #         return loss
        #     else: # inference
        #         return torch.argmax(logits, dim=2)
class TokenBERT2(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True):
        super(TokenBERT2, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.fc_linear1 = torch.nn.Linear(768, 5)

        if self.use_crf:
            self.crf1 = CRF(5, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)  # (B, L, H)
        logits_AM = self.fc_linear1(sequence_output)

        if self.use_crf:
            if labels is not None: # training
                return -self.crf1(logits_AM, labels, attention_mask.byte())
            else: # inference
                return self.crf1.decode(logits_AM, attention_mask.byte())

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs
class TokenBERT_QA(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True):
        super(TokenBERT_QA, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 300, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)

        self.fc_linear1 = torch.nn.Linear(600, 3)
        self.fc_linear2 = torch.nn.Linear(600, 2)

        self.crf1 = CRF(3, batch_first=self.batch_first)
        self.crf2 = CRF(2, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids, sent_length, pieces2word, label_mask, AM_labels=None, AS_labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #cln = self.cln(word_reps.unsqueeze(2), word_reps)

        logits_AM = self.fc_linear1(word_reps)
        logits_AS = self.fc_linear2(word_reps)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())
        #
        # sequence_output = self.dropout(word_reps)  # (B, L, H)
        # logits_AM = self.fc_linear1(word_reps)
        #
        # if self.use_crf:
        #     if AM_labels is not None: # training
        #         return -self.crf1(logits_AM, AM_labels, attention_mask.byte())
        #     else: # inference
        #         return self.crf1.decode(logits_AM, attention_mask.byte())

class CharBiLSTM(nn.Module):

    def __init__(self, device = None):
        super(CharBiLSTM, self).__init__()
        print("[Info] Building character-level LSTM")
        self.char_emb_size = 100
        self.char_size = 95
        self.dropout = nn.Dropout(0.3).to(device)
        self.hidden = 100

        # self.dropout = nn.Dropout(config.dropout).to(self.device)
        self.char_embeddings = nn.Embedding(self.char_size, self.char_emb_size)
        # self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(self.char_size, self.char_emb_size)))
        self.char_embeddings = self.char_embeddings.to(device)
        # self.char_embeddings = self.char_embeddings.to(self.device)

        self.char_lstm = nn.LSTM(self.char_emb_size, self.hidden ,num_layers=1, batch_first=True, bidirectional=False).to(device)


    # def random_embedding(self, vocab_size, embedding_dim):
    #     pretrain_emb = np.empty([vocab_size, embedding_dim])
    #     scale = np.sqrt(3.0 / embedding_dim)
    #     for index in range(vocab_size):
    #         pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
    #     return pretrain_emb

    def get_last_hiddens(self, char_seq_tensor, char_seq_len):
        """
            input:
                char_seq_tensor: (batch_size, sent_len, word_length)
                char_seq_len: (batch_size, sent_len)
            output:
                Variable(batch_size, sent_len, char_hidden_dim )
        """
        batch_size = char_seq_tensor.size(0)
        sent_len = char_seq_tensor.size(1)
        char_seq_tensor = char_seq_tensor.view(batch_size * sent_len, -1)
        char_seq_len = char_seq_len.view(batch_size * sent_len)
        sorted_seq_len, permIdx = char_seq_len.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = char_seq_tensor[permIdx]

        char_embeds = self.dropout(self.char_embeddings(sorted_seq_tensor))
        pack_input = pack_padded_sequence(char_embeds, sorted_seq_len.cpu(), batch_first=True)

        _, char_hidden = self.char_lstm(pack_input, None)  ###
        ## char_hidden = (h_t, c_t)
        #  char_hidden[0] = h_t = (2, batch_size, lstm_dimension)
        # char_rnn_out, _ = pad_packed_sequence(char_rnn_out)
        ## transpose because the first dimension is num_direction x num-layer
        hidden = char_hidden[0].transpose(1,0).contiguous().view(batch_size * sent_len, 1, -1)   ### before view, the size is ( batch_size * sent_len, 2, lstm_dimension) 2 means 2 direciton..
        return hidden[recover_idx].view(batch_size, sent_len, -1)



    def forward(self, char_input, seq_lengths):
        return self.get_last_hiddens(char_input, seq_lengths)

from typing import List, Tuple
def masked_flip(padded_sequence: torch.Tensor, sequence_lengths: List[int]) -> torch.Tensor:
    """
        Flips a padded tensor along the time dimension without affecting masked entries.
        # Parameters
        padded_sequence : `torch.Tensor`
            The tensor to flip along the time dimension.
            Assumed to be of dimensions (batch size, num timesteps, ...)
        sequence_lengths : `torch.Tensor`
            A list containing the lengths of each unpadded sequence in the batch.
        # Returns
        A `torch.Tensor` of the same shape as padded_sequence.
        """
    assert padded_sequence.size(0) == len(
        sequence_lengths
    ), f"sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}"
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [
        flipped_padded_sequence[i, num_timesteps - length :]
        for i, length in enumerate(sequence_lengths)
    ]
    sequences[0] = nn.ConstantPad1d((0, 0, 0, 85 - sequences[0].shape[0]), 0)(sequences[0])
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

import math

class MyLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(MyLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1), self.hidden_sz)
        self.all2 = nn.Linear((self.hidden_sz * 1 + self.input_sz + self.g_sz), self.hidden_sz)
        self.all3 = nn.Linear((self.hidden_sz * 1 + self.input_sz + self.g_sz), self.hidden_sz)
        self.all4 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1), self.hidden_sz)

        self.all11 = nn.Linear((self.hidden_sz * 1 + self.g_sz), self.hidden_sz)
        self.all44 = nn.Linear((self.hidden_sz * 1 + self.g_sz), self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.3)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, ht, Ct_x, mt, Ct_m):

        # # # new standard lstm
        #xt bert的输出
        #mt 图的输入
        #ht hidden
        #ct cell for xt
        #ctm cell for mt
        hx_concat = torch.cat((ht, xt), dim=1)
        hm_concat = torch.cat((ht, mt), dim=1)
        hxm_concat = torch.cat((ht, xt, mt), dim=1)

        i = self.all1(hx_concat)
        o = self.all2(hxm_concat)
        f = self.all3(hxm_concat)
        u = self.all4(hx_concat)
        ii = self.all11(hm_concat)
        uu = self.all44(hm_concat)

        i, f, o, u = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(u)
        ii, uu = torch.sigmoid(ii), torch.tanh(uu)
        Ct_x = i * u + ii * uu + f * Ct_x
        ht = o * torch.tanh(Ct_x)

        return ht, Ct_x, Ct_m

    def forward(self, x, m, init_stat=None):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        if init_stat is None:
            ht = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_x = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_m = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
        else:
            ht, Ct = init_stat
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x, Ct_m = self.node_forward(xt, ht, Ct_x, mt, Ct_m)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
            if t == 0:
                mht = ht
                mct = Ct_x
            else:
                mht = torch.max(torch.stack(hidden_seq), dim=0)[0]
                mct = torch.max(torch.stack(cell_seq), dim=0)[0]
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2)  ##batch_size x max_len x hidden
        return hidden_seq

class DepLabeledGCN(nn.Module):
    def __init__(self, device, hidden_dim, input_dim, graph_dim):
        super().__init__()
        self.lstm_hidden = hidden_dim // 2
        self.input_dim = input_dim
        self.graph_dim = graph_dim
        self.device = device
        self.gcn_layer = 2
        self.drop_lstm = nn.Dropout(0.2).to(self.device)
        self.drop_gcn = nn.Dropout(0.2).to(self.device)

        self.lstm_f = MyLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to(self.device)
        self.lstm_b = MyLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to(self.device)
        # self.lstm1 = MyLSTM(200, 100).to(self.device)
        # self.lstm_b1 = MyLSTM(200, 100).to(self.device)
        self.W = nn.ModuleList()
        for layer in range(self.gcn_layer):
            self.W.append(nn.Linear(self.graph_dim, self.graph_dim)).to(self.device)

    def forward(self, inputs, word_seq_len, adj_matrix):

        """

        :param gcn_inputs:
        :param word_seq_len:
        :param adj_matrix: should already contain the self loop
        :param dep_label_matrix:
        :return:
        """
        adj_matrix = adj_matrix.to(self.device)
        batch_size, sent_len, input_dim = inputs.size()
        denom = adj_matrix.sum(2).unsqueeze(2) + 1

        graph_input = inputs[:, :, :self.graph_dim]

        for l in range(self.gcn_layer):
            Ax = adj_matrix.bmm(graph_input)  ## N x N  times N x h  = Nxh
            AxW = self.W[l](Ax)  ## N x m
            AxW = AxW + self.W[l](graph_input)  ## self loop  N x h
            AxW = AxW / denom
            graph_input = torch.relu(AxW)

        # forward LSTM
        lstm_out = self.lstm_f(inputs, graph_input)
        # backward LSTM
        word_rep_b = masked_flip(inputs, word_seq_len.tolist())
        c_b = masked_flip(graph_input, word_seq_len.tolist())
        lstm_out_b = self.lstm_b(word_rep_b, c_b)
        lstm_out_b = masked_flip(lstm_out_b, word_seq_len.tolist())

        feature_out = torch.cat((lstm_out, lstm_out_b), dim=2)
        feature_out = self.drop_lstm(feature_out)

        return feature_out

class TokenBERT_synLSTM(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)

        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        self.gcn = DepLabeledGCN(device, 300, 500, 500)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

    def forward(self, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, AM_labels=None, AS_labels=None):
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        #word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        word_emb = torch.cat((word_emb, pos_emb), 2)
        word_emb = torch.cat((word_emb, char_features), 2)

        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        feature_out = sorted_seq_tensor

        feature_out = self.gcn(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        logits_AM = self.fc_linear1(output)
        logits_AS = self.fc_linear2(output)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())
        #
        # sequence_output = self.dropout(word_reps)  # (B, L, H)
        # logits_AM = self.fc_linear1(word_reps)
        #
        # if self.use_crf:
        #     if AM_labels is not None: # training
        #         return -self.crf1(logits_AM, AM_labels, attention_mask.byte())
        #     else: # inference
        #         return self.crf1.decode(logits_AM, attention_mask.byte())

class TokenBERT_synLSTM2(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM2, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)

        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        self.gcn = DepLabeledGCN(device, 300, 300, 300)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

    def forward(self, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, AM_labels=None, AS_labels=None):
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        #word_reps = self.dropout(word_reps)# BiLSTM
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        # word_emb = torch.cat((word_emb, pos_emb), 2)
        # word_emb = torch.cat((word_emb, char_features), 2)

        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        feature_out = sorted_seq_tensor

        feature_out = self.gcn(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        logits_AM = self.fc_linear1(output)
        logits_AS = self.fc_linear2(output)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())
        #
        # sequence_output = self.dropout(word_reps)  # (B, L, H)
        # logits_AM = self.fc_linear1(word_reps)
        #
        # if self.use_crf:
        #     if AM_labels is not None: # training
        #         return -self.crf1(logits_AM, AM_labels, attention_mask.byte())
        #     else: # inference
        #         return self.crf1.decode(logits_AM, attention_mask.byte())

class TokenBERT_synLSTM3(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM3, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)

        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        self.gcn = DepLabeledGCN(device, 300, 300, 300)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

    def forward(self, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, AM_labels=None, AS_labels=None):
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        #word_reps = self.dropout(word_reps)# BiLSTM
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        # word_emb = torch.cat((word_emb, pos_emb), 2)
        # word_emb = torch.cat((word_emb, char_features), 2)

        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        feature_out = sorted_seq_tensor

        #feature_out = self.gcn(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        logits_AM = self.fc_linear1(output)
        logits_AS = self.fc_linear2(output)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())
        #
        # sequence_output = self.dropout(word_reps)  # (B, L, H)
        # logits_AM = self.fc_linear1(word_reps)
        #
        # if self.use_crf:
        #     if AM_labels is not None: # training
        #         return -self.crf1(logits_AM, AM_labels, attention_mask.byte())
        #     else: # inference
        #         return self.crf1.decode(logits_AM, attention_mask.byte())

class TokenBERT_synLSTM4(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM4, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        self.gcn = DepLabeledGCN(device, 300, 500, 500)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=1)

        self.norm1 = LayerNorm(300)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        #word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #word_emb = torch.cat((word_emb, pos_emb), 2)
        #word_emb = torch.cat((word_emb, char_features), 2)

        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        feature_out = sorted_seq_tensor

        #feature_out = self.gcn(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'term2span':
            term2span = term2span.unsqueeze(2)
            a = term2span.cpu().numpy()
            term2span = term2span.expand(output.shape[0], output.shape[1], output.shape[1])
            b = term2span.cpu().numpy()
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += term2span
            attn = self.term_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AM = self.fc_linear_am(output3)

            if AM_labels is not None:  # training
                return -self.crf_am(logits_AM, AM_labels, label_mask.byte()), logits_AM
            else:  # inference
                return self.crf_am.decode(logits_AM, label_mask.byte()), logits_AM

        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += span2term
            attn = self.span_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output3)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class TokenBERT_synLSTM5(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM5, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        self.gcn = DepLabeledGCN(device, 300, 300, 300)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=1)

        self.norm1 = LayerNorm(300)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        #word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #word_emb = torch.cat((word_emb, pos_emb), 2)
        #word_emb = torch.cat((word_emb, char_features), 2)

        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        feature_out = sorted_seq_tensor

        feature_out = self.gcn(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'term2span':
            term2span = term2span.unsqueeze(2)
            a = term2span.cpu().numpy()
            term2span = term2span.expand(output.shape[0], output.shape[1], output.shape[1])
            b = term2span.cpu().numpy()
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += term2span
            attn = self.term_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AM = self.fc_linear_am(output3)

            if AM_labels is not None:  # training
                return -self.crf_am(logits_AM, AM_labels, label_mask.byte()), logits_AM
            else:  # inference
                return self.crf_am.decode(logits_AM, label_mask.byte()), logits_AM

        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += span2term
            attn = self.span_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output3)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

import torch.nn.functional as F
class synLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(synLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1), self.hidden_sz)
        self.all2 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1), self.hidden_sz)
        self.all3 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1), self.hidden_sz)

        self.all4 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1), self.hidden_sz)
        self.all11 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1 + self.g_sz), self.hidden_sz)
        self.all44 = nn.Linear((self.hidden_sz * 1 + self.input_sz * 1 + self.g_sz), self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.5)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, ht, Ct_x, gt):

        # # # new standard lstm
        #xt bert的输出
        #mt 图的输入
        #ht hidden
        #ct cell for xt
        #ctm cell for mt
        hx_concat = torch.cat((ht, xt), dim=1)
        hxg_concat = torch.cat((ht, xt, gt), dim=1)

        i = self.all1(hx_concat)
        o = self.all2(hx_concat)
        f = self.all3(hx_concat)

        c_ = self.all4(hx_concat)
        i_s = self.all11(hxg_concat)
        s = self.all44(hxg_concat)



        i, f, o, c_ = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(c_)
        i_s, ss = torch.sigmoid(i_s), torch.tanh(s)
        Ct_x = i * c_ + f * Ct_x
        ht_c = o * torch.tanh(Ct_x)
        ht_s = i_s * ss

        ht = ht_c + ht_s

        return ht, Ct_x

    def forward(self, x, m, init_stat=None):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        if init_stat is None:
            ht = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_x = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            #Ct_m = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
        else:
            ht, Ct = init_stat
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x = self.node_forward(xt, ht, Ct_x, mt)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
            if t == 0:
                mht = ht
                mct = Ct_x
            else:
                mht = torch.max(torch.stack(hidden_seq), dim=0)[0]
                mct = torch.max(torch.stack(cell_seq), dim=0)[0]
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2)  ##batch_size x max_len x hidden
        return hidden_seq

class GAT(nn.Module):
    def __init__(self, device, hidden_dim, input_dim, graph_dim):
        super().__init__()
        self.lstm_hidden = hidden_dim // 2
        self.input_dim = input_dim
        self.graph_dim = graph_dim
        self.device = device
        self.gcn_layer = 2
        self.drop_lstm = nn.Dropout(0.5).to(self.device)
        self.drop_gcn = nn.Dropout(0.5).to(self.device)

        self.lstm_f = synLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to(self.device)
        self.lstm_b = synLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to(self.device)
        # self.lstm1 = MyLSTM(200, 100).to(self.device)
        # self.lstm_b1 = MyLSTM(200, 100).to(self.device)
        # self.W = nn.Linear(self.graph_dim, self.graph_dim).to(self.device)


        self.W = nn.Parameter(torch.zeros(size=(graph_dim, graph_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * graph_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, inputs, word_seq_len, adj_matrix):

        """

        :param gcn_inputs:
        :param word_seq_len:
        :param adj_matrix: should already contain the self loop
        :param dep_label_matrix:
        :return:
        """
        adj_matrix = adj_matrix.to(self.device)
        batch_size, sent_len, input_dim = inputs.size()
        denom = adj_matrix.sum(2).unsqueeze(2) + 1

        graph_input = inputs[:, :, :self.graph_dim]


        h_prime = [[] for i in range(inputs.size()[0])]

        for i in range(inputs.size()[0]):
            h = torch.matmul(inputs[i], self.W)  # 32,23,600
            N = h.size()[0]  # 23
            # c = h.repeat(1, N).view(h.size()[0], N * N, -1)
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.graph_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj_matrix[i] > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            #attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime[i] = torch.matmul(attention, h).cpu().tolist()
        graph_layer1 = torch.tensor(h_prime).to('cuda')

        # for l in range(self.gcn_layer):
        #     Ax = adj_matrix.bmm(graph_input)  ## N x N  times N x h  = Nxh
        #     AxW = self.W[l](Ax)  ## N x m
        #     AxW = AxW + self.W[l](graph_input)  ## self loop  N x h
        #     AxW = AxW / denom
        #     graph_input = torch.relu(AxW)

        # forward LSTM
        lstm_out = self.lstm_f(inputs, graph_layer1)
        # backward LSTM
        word_rep_b = masked_flip(inputs, word_seq_len.tolist())
        c_b = masked_flip(graph_layer1, word_seq_len.tolist())
        lstm_out_b = self.lstm_b(word_rep_b, c_b)
        lstm_out_b = masked_flip(lstm_out_b, word_seq_len.tolist())

        feature_out = torch.cat((lstm_out, lstm_out_b), dim=2)
        feature_out = self.drop_lstm(feature_out)

        return feature_out


class TokenBERT_synLSTM6(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM6, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.gcn = GAT(device, 300, 300, 300)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=1)

        self.norm1 = LayerNorm(300)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # word_emb = torch.cat((word_reps, pos_emb), 2)
        # word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)



        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        feature_out = sorted_seq_tensor

        feature_out = self.gcn(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'term2span':
            term2span = term2span.unsqueeze(2)
            a = term2span.cpu().numpy()
            term2span = term2span.expand(output.shape[0], output.shape[1], output.shape[1])
            b = term2span.cpu().numpy()
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += term2span
            attn = self.term_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AM = self.fc_linear_am(output3)

            if AM_labels is not None:  # training
                return -self.crf_am(logits_AM, AM_labels, label_mask.byte()), logits_AM
            else:  # inference
                return self.crf_am.decode(logits_AM, label_mask.byte()), logits_AM

        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += span2term
            attn = self.span_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output3)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class TokenBERT_synLSTM7(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM7, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(400, 7)
        self.fc_linear2 = torch.nn.Linear(400, 4)

        self.fc_linear_am = torch.nn.Linear(400, 7)
        self.fc_linear_as = torch.nn.Linear(400, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.gcn = GAT(device, 300, 300, 300)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.5).to(device)

        self.W_q = nn.Linear(400, 400, bias=True)
        self.W_k = nn.Linear(400, 400, bias=True)
        self.W_v = nn.Linear(400, 400, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=1)

        self.norm1 = LayerNorm(400)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # word_emb = torch.cat((word_reps, pos_emb), 2)
        # word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)



        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        feature_out = sorted_seq_tensor

        feature_out = self.gcn(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        output = torch.cat((output, pos_emb), 2)
        #output = torch.cat((output, char_features), 2)

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'term2span':
            term2span = term2span.unsqueeze(2)
            a = term2span.cpu().numpy()
            term2span = term2span.expand(output.shape[0], output.shape[1], output.shape[1])
            b = term2span.cpu().numpy()
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += term2span
            attn = self.term_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AM = self.fc_linear_am(output3)

            if AM_labels is not None:  # training
                return -self.crf_am(logits_AM, AM_labels, label_mask.byte()), logits_AM
            else:  # inference
                return self.crf_am.decode(logits_AM, label_mask.byte()), logits_AM

        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += span2term
            attn = self.span_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output3)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class GAT2layer(nn.Module):
    def __init__(self, device, hidden_dim, input_dim, graph_dim):
        super().__init__()
        self.lstm_hidden = hidden_dim // 2
        self.input_dim = input_dim
        self.graph_dim = graph_dim
        self.device = device
        self.gcn_layer = 2
        self.drop_lstm = nn.Dropout(0.5).to(self.device)
        self.drop_gcn = nn.Dropout(0.5).to(self.device)

        self.lstm_f = synLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to(self.device)
        self.lstm_b = synLSTM(self.input_dim, self.lstm_hidden, self.graph_dim).to(self.device)
        # self.lstm1 = MyLSTM(200, 100).to(self.device)
        # self.lstm_b1 = MyLSTM(200, 100).to(self.device)
        # self.W = nn.Linear(self.graph_dim, self.graph_dim).to(self.device)


        self.W = nn.Parameter(torch.zeros(size=(graph_dim, graph_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * graph_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.W2 = nn.Parameter(torch.zeros(size=(graph_dim, graph_dim)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(2 * graph_dim, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

    def forward(self, inputs, word_seq_len, adj_matrix):

        """

        :param gcn_inputs:
        :param word_seq_len:
        :param adj_matrix: should already contain the self loop
        :param dep_label_matrix:
        :return:
        """
        adj_matrix = adj_matrix.to(self.device)
        batch_size, sent_len, input_dim = inputs.size()
        denom = adj_matrix.sum(2).unsqueeze(2) + 1

        graph_input = inputs[:, :, :self.graph_dim]


        h_prime = [[] for i in range(inputs.size()[0])]

        for i in range(inputs.size()[0]):
            h = torch.matmul(inputs[i], self.W)  # 32,23,600
            N = h.size()[0]  # 23
            # c = h.repeat(1, N).view(h.size()[0], N * N, -1)
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.graph_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj_matrix[i] > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            #attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime[i] = torch.matmul(attention, h).cpu().tolist()
        graph_layer1 = torch.tensor(h_prime).to('cuda')

        h_prime2 = [[] for i in range(graph_layer1.size()[0])]

        for i in range(graph_layer1.size()[0]):
            h = torch.matmul(graph_layer1[i], self.W2)  # 32,23,600
            N = h.size()[0]  # 23
            # c = h.repeat(1, N).view(h.size()[0], N * N, -1)
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.graph_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a2).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj_matrix[i] > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            # attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime2[i] = torch.matmul(attention, h).cpu().tolist()
        graph_layer2 = torch.tensor(h_prime2).to('cuda')

        # for l in range(self.gcn_layer):
        #     Ax = adj_matrix.bmm(graph_input)  ## N x N  times N x h  = Nxh
        #     AxW = self.W[l](Ax)  ## N x m
        #     AxW = AxW + self.W[l](graph_input)  ## self loop  N x h
        #     AxW = AxW / denom
        #     graph_input = torch.relu(AxW)

        # forward LSTM
        lstm_out = self.lstm_f(inputs, graph_layer1)
        # backward LSTM
        word_rep_b = masked_flip(inputs, word_seq_len.tolist())
        c_b = masked_flip(graph_layer1, word_seq_len.tolist())
        lstm_out_b = self.lstm_b(word_rep_b, c_b)
        lstm_out_b = masked_flip(lstm_out_b, word_seq_len.tolist())

        feature_out = torch.cat((lstm_out, lstm_out_b), dim=2)
        feature_out = self.drop_lstm(feature_out)

        lstm_out = self.lstm_f(feature_out, graph_layer2)
        # backward LSTM
        word_rep_b = masked_flip(feature_out, word_seq_len.tolist())
        c_b = masked_flip(graph_layer2, word_seq_len.tolist())
        lstm_out_b = self.lstm_b(word_rep_b, c_b)
        lstm_out_b = masked_flip(lstm_out_b, word_seq_len.tolist())

        feature_out = torch.cat((lstm_out, lstm_out_b), dim=2)
        feature_out = self.drop_lstm(feature_out)

        return feature_out

class TokenBERT_synLSTM8(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM8, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(968, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.gcn = GAT2layer(device, 300, 300, 300)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.5).to(device)

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=1)

        self.norm1 = LayerNorm(300)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_emb = torch.cat((word_reps, pos_emb), 2)
        word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_emb, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)



        word_rep = self.word_drop(word_emb)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        feature_out = sorted_seq_tensor

        feature_out = self.gcn(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'term2span':
            term2span = term2span.unsqueeze(2)
            a = term2span.cpu().numpy()
            term2span = term2span.expand(output.shape[0], output.shape[1], output.shape[1])
            b = term2span.cpu().numpy()
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += term2span
            attn = self.term_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AM = self.fc_linear_am(output3)

            if AM_labels is not None:  # training
                return -self.crf_am(logits_AM, AM_labels, label_mask.byte()), logits_AM
            else:  # inference
                return self.crf_am.decode(logits_AM, label_mask.byte()), logits_AM

        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += span2term
            attn = self.span_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output3)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class TokenBERT_LSTM_CRF(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_LSTM_CRF, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder1 = nn.LSTM(768, 150, num_layers=1, batch_first=True, bidirectional=True)
        self.encoder2 = nn.LSTM(768, 150, num_layers=1, batch_first=True, bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)


        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)


    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder1(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)



        word_rep = self.dropout(word_emb)
        logits_AM = self.fc_linear1(word_rep)



        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder2(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        word_rep = self.dropout(word_emb)
        logits_AS = self.fc_linear2(word_rep)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())

class TokenBERT_synLSTM_CRF(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_synLSTM_CRF, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder1 = nn.LSTM(768, 150, num_layers=1, batch_first=True, bidirectional=True)
        self.encoder2 = nn.LSTM(768, 150, num_layers=1, batch_first=True, bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.gcn1 = GAT(device, 300, 768, 768)  ### lstm hidden size
        self.gcn2 = GAT(device, 300, 768, 768)  ### lstm hidden size

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)


    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)


        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_reps[permIdx]

        feature_out = sorted_seq_tensor

        feature_out = self.gcn1(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        output = self.dropout(output)
        logits_AM = self.fc_linear1(output)

        sorted_seq_len, permIdx = sent_length.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_reps[permIdx]

        feature_out = sorted_seq_tensor

        feature_out = self.gcn2(feature_out, sent_length, graph[permIdx])

        output = feature_out[recover_idx]

        output = self.dropout(output)
        logits_AS = self.fc_linear2(output)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())



import numpy as np
import torch.autograd as autograd

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, use_glu, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.use_glu = use_glu

        self.w_qs = nn.Linear(d_model, n_head * d_k).cuda()
        self.w_ks = nn.Linear(d_model, n_head * d_k).cuda()
        self.w_vs = nn.Linear(d_model, n_head * d_v).cuda()
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model).cuda()

        self.fc = nn.Linear(n_head * d_v, d_model).cuda()
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout).cuda()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(F.relu(self.fc(output)))

        if self.use_glu == 1:
            output = self.layer_norm(residual+output)
        elif self.use_glu == 2:
            act = self.sigmoid
            gated = act(self.w1(residual))
            output = self.layer_norm((1 - gated) * residual + output + gated * self.w2(residual))
        elif self.use_glu == 3:
            act = self.sigmoid
            gated = act(self.w1(residual))
            output = residual + self.layer_norm((1 - gated) * output + gated * self.w2(residual))

        return output, attn
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + autograd.Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
class StarEncoderLayer(nn.Module):
    ''' Star-Transformer: https://arxiv.org/pdf/1902.09113v2.pdf '''

    def __init__(self, d_model, n_head, d_k, d_v, glu_type, dropout):
        super(StarEncoderLayer, self).__init__()
        self.slf_attn_satellite = MultiHeadAttention(
            n_head, d_model, d_k, d_v, use_glu=glu_type, dropout=dropout)
        self.slf_attn_relay = MultiHeadAttention(
            n_head, d_model, d_k, d_v, use_glu=glu_type, dropout=dropout)

    def forward(self, h, e, s, non_pad_mask=None, slf_attn_mask=None):
        # satellite node
        batch_size, seq_len, d_model = h.size()
        h_extand = torch.zeros(batch_size, seq_len+2, d_model, dtype=torch.float, device=h.device)
        h_extand[:, 1:seq_len+1, :] = h  # head and tail padding(not cycle)
        s = s.reshape([batch_size, 1, d_model])
        s_expand = s.expand([batch_size, seq_len, d_model])
        context = torch.cat((h_extand[:, 0:seq_len, :],
                             h_extand[:, 1:seq_len+1, :],
                             h_extand[:, 2:seq_len+2, :],
                             e,
                             s_expand),
                            2)
        context = context.reshape([batch_size*seq_len, 5, d_model])
        h = h.reshape([batch_size*seq_len, 1, d_model])

        h, _ = self.slf_attn_satellite(h, context, context, mask=slf_attn_mask)
        h = torch.squeeze(h, 1).reshape([batch_size, seq_len, d_model])
        if non_pad_mask is not None:
            h *= non_pad_mask

        # virtual relay node
        s_h = torch.cat((s, h), 1)
        s, _ = self.slf_attn_relay(
            s, s_h, s_h, mask=slf_attn_mask)
        s = torch.squeeze(s, 1)

        return h, s
class TokenBERT_sequence(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_sequence, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)


        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)


    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)



        word_rep = self.dropout(word_emb)


        logits_AM = self.fc_linear1(word_rep)
        logits_AS = self.fc_linear2(word_rep)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())

class TokenBERT_LSTM_CRF_start_T(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_LSTM_CRF_start_T, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder1 = nn.LSTM(768, 150, num_layers=1, batch_first=True, bidirectional=True)
        self.encoder2 = nn.LSTM(768, 150, num_layers=1, batch_first=True, bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        #self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)


        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(768, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.word2star2 = nn.Linear(768, 300)
        self.posi2 = PositionalEncoding(300, 0.5)
        self.star_transformer2 = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )
        # self.word2star = self.word2star.cuda()
        # self.posi = self.posi.cuda()
        # self.star_transformer = self.star_transformer.cuda()

        self.heads_layer = nn.GRU(768, 150, num_layers= 1, batch_first=True,
                                  bidirectional=True)
        self.tails_layer = nn.GRU(768, 150, num_layers= 1, batch_first=True,
                                  bidirectional=True)

        self.hidden2head = nn.Linear(300, 2)
        self.hidden2tail = nn.Linear(300, 2)

        self.drophead = nn.Dropout(0.5)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)


    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, batch_head=None, batch_tail=None, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        x = self.word2star(word_reps)
        h = self.posi(x)
        s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        for idx in range(self.HP_star_layer):
            h, s = self.star_transformer(h, x, s)
        feature_out1 = h



        x = self.word2star2(word_reps)
        h = self.posi2(x)
        s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        for idx in range(self.HP_star_layer):
            h, s = self.star_transformer2(h, x, s)
        feature_out = h

        output = self.dropout(feature_out)
        logits_AS = self.fc_linear2(output)

        # heads_out, hidden = self.heads_layer(word_reps)
        # heads_feature = self.drophead(heads_out)  # heads_out (batch_size, seq_len, hidden_size)
        # heads_outputs = self.hidden2head(heads_feature)
        # # extract tail feature
        # tails_out, hidden = self.tails_layer(word_reps)
        # tails_feature = self.drophead(tails_out)  # tails_out (batch_size, seq_len, hidden_size)
        # tails_outputs = self.hidden2tail(tails_feature)
        #
        # feature_out1 = self.dropout(feature_out1)
        # outputs1 = feature_out1 + heads_feature +  tails_feature
        feature_out1 = self.dropout(feature_out1)
        #outputs = self.weight1 * feature_out1 + self.weight2 * heads_feature + self.weight3 * tails_feature
        logits_AM = self.fc_linear1(feature_out1)



        if AM_labels is not None:  # training
            # batch_size = word_reps.size(0)
            # seq_len = word_reps.size(1)
            # head_loss_function = nn.CrossEntropyLoss()
            # heads_out = heads_outputs.view(batch_size * seq_len, -1)
            # head_loss = head_loss_function(heads_out, batch_head.view(batch_size * seq_len))
            # # tail loss
            # tail_loss_function = nn.CrossEntropyLoss()
            # tails_out = tails_outputs.view(batch_size * seq_len, -1)
            # tail_loss = tail_loss_function(tails_out, batch_tail.view(batch_size * seq_len))

            #return -self.crf1(logits_AM, AM_labels, label_mask.byte(), reduction= 'token_mean') - self.crf2(logits_AS, AS_labels, label_mask.byte(), reduction= 'token_mean'), head_loss, tail_loss
            return -self.crf1(logits_AM, AM_labels, label_mask.byte(), reduction= 'token_mean') - self.crf2(logits_AS, AS_labels, label_mask.byte(), reduction= 'token_mean')
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # h = self.W(input)
        # # [batch_size, N, out_features]
        # # batch_size, N, _ = h.size()
        # middle_result1 = torch.matmul(h, self.a1)
        # middle_result2 = torch.matmul(h, self.a2).transpose(0, 1)
        # #dots = torch.matmul(h, h.transpose(0, 1))
        # e = self.leakyrelu(middle_result1 + middle_result2)
        # attention = e.masked_fill(adj == 0, -1e9)
        # attention = F.softmax(attention, dim=1)
        # atte = attention.cpu().tolist()
        h = self.W(input)
        N = h.size()[0]  # 23
        # c = h.repeat(1, N).view(h.size()[0], N * N, -1)
        middle_result1 = torch.matmul(h, self.a1)
        middle_result2 = torch.matmul(h, self.a2).transpose(0, 1)
        #a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(middle_result1 + middle_result2)
        #e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)



        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        h_prime = input + h_prime
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layer):
        """
        :param nfeat:   输入的维度
        :param nhid:   graph内部的维度
        :param nclass:    label_alphabet.size()+1
        :param dropout:
        :param alpha:
        :param nheads:
        :param layer:
        """
        super(GAT2, self).__init__()
        self.dropout = dropout
        self.layer = layer
        if self.layer == 1:
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)]
        else:
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=0.1, alpha=alpha, concat=True) for _ in
                               range(nheads)]
            # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout)
        if self.layer == 1:
            x = torch.stack([att(x, adj) for att in self.attentions], dim=1)
            x = x.sum(1)
            x = F.dropout(x, self.dropout)
            return F.elu(x)
        else:
            x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # [batch, max_seq_len, nhid * nheads]
            x = F.dropout(x, self.dropout)
            # x = F.elu(self.out_att(x, adj))
            return F.elu(x)

class TokenBERT_1117(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_1117, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 300, 300)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(300, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=1)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer= 1)
        self.stance_classifier = nn.Linear(300, 2)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None, starts=None, ends=None, topic_indices=None, label_stance=None, token_spacy_ids= None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        # a1 = pieces2word
        # a2 = pieces2word.eq(0)
        # a3 = pieces2word.eq(0).unsqueeze(-1)
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # word_emb = torch.cat((word_reps, pos_emb), 2)
        # word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        word_rep = self.word_drop(word_emb)
        #syn_lstm
        feature_out = self.syn_lstm(word_rep, sent_length, graph)

        #star_transformer
        x = self.word2star(feature_out)
        h = self.posi(x)
        s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        for idx in range(self.HP_star_layer):
            h, s = self.star_transformer(h, x, s)

        output = h

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)
            logits_AM = F.softmax(logits_AM, dim=-1)
            logits_AS = F.softmax(logits_AS, dim=-1)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'stance':

            #last_hidden_state_cls = torch.concat((last_hidden_state_cls, token), dim=0)
            topic_ = self.embed(topic_indices)
            topic_label = []
            span_embs = None
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    hi=word_rep[i][starts[i][j]]
                    hj=word_rep[i][ends[i][j]]
                    topici=topic_[i]
                    if span_embs == None:
                        span_embs=(hi+hj+topici).view(1, -1)
                    else:
                        span_embs = torch.concat((span_embs, (hi+hj+topici).view(1, -1)), dim=0)
                    topic_label.append(topic_indices[i])

            topic_label = torch.Tensor(topic_label).float().to(self.device).view(-1, 1)

            #token_spacy_ids = []
            token_embs = []
            input_is = []
            # token_token
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    input_i = token_spacy_ids[i][starts[i][j]:ends[i][j]+1]
                    token = word_rep[i][starts[i][j]:ends[i][j]+1, :]
                    input_is.extend(input_i)
                    token_embs.append(token)
                    span_embs = torch.concat((span_embs, token), dim=0)

            input_is = torch.Tensor(input_is).float().to(self.device).view(-1, 1)
            # labels = input_is.contiguous().view(-1, 1)
            tt_adj = torch.eq(input_is, input_is.T).float().to(self.device)
            # from scipy import sparse
            # import numpy as np
            # X_csr = sparse.csr_matrix(tt_adj.cpu().numpy())
            # print(X_csr)

            # sent_token
            st_adj = np.zeros((topic_label.shape[0], input_is.shape[0]))
            start = 0
            span_count = 0
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    st_adj[span_count][start:ends[i][j] - starts[i][j] + 1 +start] = 1
                    start = start + (ends[i][j] - starts[i][j] + 1)
                    span_count+=1
                # print(st_adj[i])

            # token_sent
            ts_adj = st_adj.T

            st_adj = torch.Tensor(st_adj).float().to(self.device)
            ts_adj = torch.Tensor(ts_adj).float().to(self.device)

            # sent_sent
            topic_label = topic_label.contiguous().view(-1, 1)
            ss_adj = torch.eq(topic_label, topic_label.T).float().to(self.device)  # 逐元素的比较，若相同位置的两个元素相同，则返回True

            # adj
            adj = np.zeros((topic_label.shape[0] + input_is.shape[0], topic_label.shape[0] + input_is.shape[0]))
            adj = torch.Tensor(adj).float().to(self.device)
            adj[:topic_label.shape[0], :topic_label.shape[0]] = ss_adj
            adj[:topic_label.shape[0], topic_label.shape[0]:] = st_adj
            adj[topic_label.shape[0]:, :topic_label.shape[0]] = ts_adj
            adj[topic_label.shape[0]:, topic_label.shape[0]:] = tt_adj

            aaa = sum(sum(adj - adj.T))
            a = adj.cpu().numpy()
            gcn_out = self.gat(span_embs, adj)

            stance_out = gcn_out[:topic_label.shape[0], :]

            logits = self.stance_classifier(stance_out)
            logits = F.softmax(logits, dim=-1)
            if label_stance is not None:  # training
                label_ = []
                for i in label_stance:
                    for j in i:
                        label_.append(j)
                label_ = torch.Tensor(label_).to(self.device).view(-1, 1)
                label_ = label_.to(torch.int64)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, 2),
                    label_.view(-1)
                )
                return loss, logits
            else: # inference
                return torch.argmax(logits, dim=-1), logits


        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += span2term
            attn = self.span_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            #output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output2)
            logits_AS = F.softmax(logits_AS, dim=-1)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':
            logits_AM2 = torch.Tensor(logits_AM2).to(self.device)

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())


class TokenBERT_1121(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_1121, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc_linear1 = torch.nn.Linear(768, 5)
        self.cln = LayerNorm(600, 600, conditional=True)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 768, 768)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(300, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=1)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer= 1)
        self.stance_classifier = nn.Linear(300, 2)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None, starts=None, ends=None, topic_indices=None, label_stance=None, token_spacy_ids= None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        #得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # word_emb = torch.cat((word_reps, pos_emb), 2)
        # word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        #LSTM
        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #syn_lstm
        feature_out = self.syn_lstm(word_reps, sent_length, graph)
        word_rep = self.word_drop(feature_out)

        #star_transformer
        x = self.word2star(feature_out)
        h = self.posi(x)
        s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        for idx in range(self.HP_star_layer):
            h, s = self.star_transformer(h, x, s)

        output = h

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)
            logits_AM = F.softmax(logits_AM, dim=-1)
            logits_AS = F.softmax(logits_AS, dim=-1)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'stance':

            #last_hidden_state_cls = torch.concat((last_hidden_state_cls, token), dim=0)
            topic_ = self.embed(topic_indices)
            topic_label = []
            span_embs = None
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    hi=output[i][starts[i][j]]
                    hj=output[i][ends[i][j]]
                    topici=topic_[i]
                    if span_embs == None:
                        span_embs=(hi+hj+topici).view(1, -1)
                    else:
                        span_embs = torch.concat((span_embs, (hi+hj+topici).view(1, -1)), dim=0)
                    topic_label.append(topic_indices[i])

            topic_label = torch.Tensor(topic_label).float().to(self.device).view(-1, 1)

            #token_spacy_ids = []
            token_embs = []
            input_is = []
            # token_token
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    input_i = token_spacy_ids[i][starts[i][j]:ends[i][j]+1]
                    token = output[i][starts[i][j]:ends[i][j]+1, :]
                    input_is.extend(input_i)
                    token_embs.append(token)
                    span_embs = torch.concat((span_embs, token), dim=0)

            input_is = torch.Tensor(input_is).float().to(self.device).view(-1, 1)
            # labels = input_is.contiguous().view(-1, 1)
            tt_adj = torch.eq(input_is, input_is.T).float().to(self.device)
            # from scipy import sparse
            # import numpy as np
            # X_csr = sparse.csr_matrix(tt_adj.cpu().numpy())
            # print(X_csr)

            # sent_token
            st_adj = np.zeros((topic_label.shape[0], input_is.shape[0]))
            start = 0
            span_count = 0
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    st_adj[span_count][start:ends[i][j] - starts[i][j] + 1 +start] = 1
                    start = start + (ends[i][j] - starts[i][j] + 1)
                    span_count+=1
                # print(st_adj[i])

            # token_sent
            ts_adj = st_adj.T

            st_adj = torch.Tensor(st_adj).float().to(self.device)
            ts_adj = torch.Tensor(ts_adj).float().to(self.device)

            # sent_sent
            topic_label = topic_label.contiguous().view(-1, 1)
            ss_adj = torch.eq(topic_label, topic_label.T).float().to(self.device)  # 逐元素的比较，若相同位置的两个元素相同，则返回True

            # adj
            adj = np.zeros((topic_label.shape[0] + input_is.shape[0], topic_label.shape[0] + input_is.shape[0]))
            adj = torch.Tensor(adj).float().to(self.device)
            adj[:topic_label.shape[0], :topic_label.shape[0]] = ss_adj
            adj[:topic_label.shape[0], topic_label.shape[0]:] = st_adj
            adj[topic_label.shape[0]:, :topic_label.shape[0]] = ts_adj
            adj[topic_label.shape[0]:, topic_label.shape[0]:] = tt_adj

            aaa = sum(sum(adj - adj.T))
            a = adj.cpu().numpy()
            gcn_out = self.gat(span_embs, adj)

            stance_out = gcn_out[:topic_label.shape[0], :]

            logits = self.stance_classifier(stance_out)
            logits = F.softmax(logits, dim=-1)
            if label_stance is not None:  # training
                label_ = []
                for i in label_stance:
                    for j in i:
                        label_.append(j)
                label_ = torch.Tensor(label_).to(self.device).view(-1, 1)
                label_ = label_.to(torch.int64)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, 2),
                    label_.view(-1)
                )
                return loss, logits
            else: # inference
                return torch.argmax(logits, dim=-1), logits


        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn += span2term
            attn = self.span_softmax(attn)
            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn.transpose(1, 2), values)

            #output3 = output + output2
            #output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output2)
            logits_AS = F.softmax(logits_AS, dim=-1)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':
            logits_AM2 = torch.Tensor(logits_AM2).to(self.device)

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())


class TokenBERT_1122(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_1122, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        #self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 968, 968)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.4).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(300, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=2)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer= 1)
        self.stance_classifier = nn.Linear(300, 2)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None, starts=None, ends=None, topic_indices=None, label_stance=None, token_spacy_ids= None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        #得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_emb = torch.cat((word_reps, pos_emb), 2)
        word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        #LSTM
        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #syn_lstm
        feature_out = self.syn_lstm(word_emb, sent_length, graph)
        #word_rep = self.word_drop(feature_out)

        #star_transformer
        x = self.word2star(feature_out)
        h = self.posi(x)
        s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        for idx in range(self.HP_star_layer):
            h, s = self.star_transformer(h, x, s)

        output = h

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)
            logits_AM = F.softmax(logits_AM, dim=-1)
            logits_AS = F.softmax(logits_AS, dim=-1)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'stance':

            #last_hidden_state_cls = torch.concat((last_hidden_state_cls, token), dim=0)
            topic_ = self.embed(topic_indices)
            topic_label = []
            span_embs = None
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    hi=output[i][starts[i][j]]
                    hj=output[i][ends[i][j]]
                    topici=topic_[i]
                    if span_embs == None:
                        span_embs=(hi+hj+topici).view(1, -1)
                    else:
                        span_embs = torch.concat((span_embs, (hi+hj+topici).view(1, -1)), dim=0)
                    topic_label.append(topic_indices[i])

            topic_label = torch.Tensor(topic_label).float().to(self.device).view(-1, 1)

            #token_spacy_ids = []
            token_embs = []
            input_is = []
            # token_token
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    input_i = token_spacy_ids[i][starts[i][j]:ends[i][j]+1]
                    token = output[i][starts[i][j]:ends[i][j]+1, :]
                    input_is.extend(input_i)
                    token_embs.append(token)
                    span_embs = torch.concat((span_embs, token), dim=0)

            input_is = torch.Tensor(input_is).float().to(self.device).view(-1, 1)
            # labels = input_is.contiguous().view(-1, 1)
            tt_adj = torch.eq(input_is, input_is.T).float().to(self.device)
            # from scipy import sparse
            # import numpy as np
            # X_csr = sparse.csr_matrix(tt_adj.cpu().numpy())
            # print(X_csr)

            # sent_token
            st_adj = np.zeros((topic_label.shape[0], input_is.shape[0]))
            start = 0
            span_count = 0
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    st_adj[span_count][start:ends[i][j] - starts[i][j] + 1 +start] = 1
                    start = start + (ends[i][j] - starts[i][j] + 1)
                    span_count+=1
                # print(st_adj[i])

            # token_sent
            ts_adj = st_adj.T

            st_adj = torch.Tensor(st_adj).float().to(self.device)
            ts_adj = torch.Tensor(ts_adj).float().to(self.device)

            # sent_sent
            topic_label = topic_label.contiguous().view(-1, 1)
            ss_adj = torch.eq(topic_label, topic_label.T).float().to(self.device)  # 逐元素的比较，若相同位置的两个元素相同，则返回True

            # adj
            adj = np.zeros((topic_label.shape[0] + input_is.shape[0], topic_label.shape[0] + input_is.shape[0]))
            adj = torch.Tensor(adj).float().to(self.device)
            adj[:topic_label.shape[0], :topic_label.shape[0]] = ss_adj
            adj[:topic_label.shape[0], topic_label.shape[0]:] = st_adj
            adj[topic_label.shape[0]:, :topic_label.shape[0]] = ts_adj
            adj[topic_label.shape[0]:, topic_label.shape[0]:] = tt_adj

            aaa = sum(sum(adj - adj.T))
            a = adj.cpu().numpy()
            gcn_out = self.gat(span_embs, adj)

            stance_out = gcn_out[:topic_label.shape[0], :]

            logits = self.stance_classifier(stance_out)
            logits = F.softmax(logits, dim=-1)
            if label_stance is not None:  # training
                label_ = []
                for i in label_stance:
                    for j in i:
                        label_.append(j)
                label_ = torch.Tensor(label_).to(self.device).view(-1, 1)
                label_ = label_.to(torch.int64)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, 2),
                    label_.view(-1)
                )
                return loss, logits
            else: # inference
                return torch.argmax(logits, dim=-1), logits


        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            # span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            # b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn = self.span_softmax(attn)
            attn = attn * span2term

            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn, values)

            #output3 = output + output2
            #output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output2)
            logits_AS = F.softmax(logits_AS, dim=-1)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':
            logits_AM2 = torch.Tensor(logits_AM2).to(self.device)

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class TokenBERT_1122_nsyn(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_1122_nsyn, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        #self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 968, 968)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(968, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=2)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer= 1)
        self.stance_classifier = nn.Linear(300, 2)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None, starts=None, ends=None, topic_indices=None, label_stance=None, token_spacy_ids= None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        #得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_emb = torch.cat((word_reps, pos_emb), 2)
        word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        #LSTM
        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #syn_lstm
        #feature_out = self.syn_lstm(word_emb, sent_length, graph)
        #word_rep = self.word_drop(feature_out)

        #star_transformer
        x = self.word2star(word_emb)
        h = self.posi(x)
        s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        for idx in range(self.HP_star_layer):
            h, s = self.star_transformer(h, x, s)

        output = h

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)
            logits_AM = F.softmax(logits_AM, dim=-1)
            logits_AS = F.softmax(logits_AS, dim=-1)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'stance':

            #last_hidden_state_cls = torch.concat((last_hidden_state_cls, token), dim=0)
            topic_ = self.embed(topic_indices)
            topic_label = []
            span_embs = None
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    hi=output[i][starts[i][j]]
                    hj=output[i][ends[i][j]]
                    topici=topic_[i]
                    if span_embs == None:
                        span_embs=(hi+hj+topici).view(1, -1)
                    else:
                        span_embs = torch.concat((span_embs, (hi+hj+topici).view(1, -1)), dim=0)
                    topic_label.append(topic_indices[i])

            topic_label = torch.Tensor(topic_label).float().to(self.device).view(-1, 1)

            #token_spacy_ids = []
            token_embs = []
            input_is = []
            # token_token
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    input_i = token_spacy_ids[i][starts[i][j]:ends[i][j]+1]
                    token = output[i][starts[i][j]:ends[i][j]+1, :]
                    input_is.extend(input_i)
                    token_embs.append(token)
                    span_embs = torch.concat((span_embs, token), dim=0)

            input_is = torch.Tensor(input_is).float().to(self.device).view(-1, 1)
            # labels = input_is.contiguous().view(-1, 1)
            tt_adj = torch.eq(input_is, input_is.T).float().to(self.device)
            # from scipy import sparse
            # import numpy as np
            # X_csr = sparse.csr_matrix(tt_adj.cpu().numpy())
            # print(X_csr)

            # sent_token
            st_adj = np.zeros((topic_label.shape[0], input_is.shape[0]))
            start = 0
            span_count = 0
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    st_adj[span_count][start:ends[i][j] - starts[i][j] + 1 +start] = 1
                    start = start + (ends[i][j] - starts[i][j] + 1)
                    span_count+=1
                # print(st_adj[i])

            # token_sent
            ts_adj = st_adj.T

            st_adj = torch.Tensor(st_adj).float().to(self.device)
            ts_adj = torch.Tensor(ts_adj).float().to(self.device)

            # sent_sent
            topic_label = topic_label.contiguous().view(-1, 1)
            ss_adj = torch.eq(topic_label, topic_label.T).float().to(self.device)  # 逐元素的比较，若相同位置的两个元素相同，则返回True

            # adj
            adj = np.zeros((topic_label.shape[0] + input_is.shape[0], topic_label.shape[0] + input_is.shape[0]))
            adj = torch.Tensor(adj).float().to(self.device)
            adj[:topic_label.shape[0], :topic_label.shape[0]] = ss_adj
            adj[:topic_label.shape[0], topic_label.shape[0]:] = st_adj
            adj[topic_label.shape[0]:, :topic_label.shape[0]] = ts_adj
            adj[topic_label.shape[0]:, topic_label.shape[0]:] = tt_adj

            aaa = sum(sum(adj - adj.T))
            a = adj.cpu().numpy()
            gcn_out = self.gat(span_embs, adj)

            stance_out = gcn_out[:topic_label.shape[0], :]

            logits = self.stance_classifier(stance_out)
            logits = F.softmax(logits, dim=-1)
            if label_stance is not None:  # training
                label_ = []
                for i in label_stance:
                    for j in i:
                        label_.append(j)
                label_ = torch.Tensor(label_).to(self.device).view(-1, 1)
                label_ = label_.to(torch.int64)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, 2),
                    label_.view(-1)
                )
                return loss, logits
            else: # inference
                return torch.argmax(logits, dim=-1), logits


        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            # span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            # b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn = self.span_softmax(attn)
            attn = attn * span2term

            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn, values)

            #output3 = output + output2
            #output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output2)
            logits_AS = F.softmax(logits_AS, dim=-1)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':
            logits_AM2 = torch.Tensor(logits_AM2).to(self.device)

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class TokenBERT_1122_nstar(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_1122_nstar, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        #self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 968, 968)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(300, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=2)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer= 1)
        self.stance_classifier = nn.Linear(300, 2)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None, starts=None, ends=None, topic_indices=None, label_stance=None, token_spacy_ids= None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        #得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_emb = torch.cat((word_reps, pos_emb), 2)
        word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        #LSTM
        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #syn_lstm
        feature_out = self.syn_lstm(word_emb, sent_length, graph)
        #word_rep = self.word_drop(feature_out)

        #star_transformer
        # x = self.word2star(feature_out)
        # h = self.posi(x)
        # s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        # for idx in range(self.HP_star_layer):
        #     h, s = self.star_transformer(h, x, s)

        output = feature_out

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)
            logits_AM = F.softmax(logits_AM, dim=-1)
            logits_AS = F.softmax(logits_AS, dim=-1)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'stance':

            #last_hidden_state_cls = torch.concat((last_hidden_state_cls, token), dim=0)
            topic_ = self.embed(topic_indices)
            topic_label = []
            span_embs = None
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    hi=output[i][starts[i][j]]
                    hj=output[i][ends[i][j]]
                    topici=topic_[i]
                    if span_embs == None:
                        span_embs=(hi+hj+topici).view(1, -1)
                    else:
                        span_embs = torch.concat((span_embs, (hi+hj+topici).view(1, -1)), dim=0)
                    topic_label.append(topic_indices[i])

            topic_label = torch.Tensor(topic_label).float().to(self.device).view(-1, 1)

            #token_spacy_ids = []
            token_embs = []
            input_is = []
            # token_token
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    input_i = token_spacy_ids[i][starts[i][j]:ends[i][j]+1]
                    token = output[i][starts[i][j]:ends[i][j]+1, :]
                    input_is.extend(input_i)
                    token_embs.append(token)
                    span_embs = torch.concat((span_embs, token), dim=0)

            input_is = torch.Tensor(input_is).float().to(self.device).view(-1, 1)
            # labels = input_is.contiguous().view(-1, 1)
            tt_adj = torch.eq(input_is, input_is.T).float().to(self.device)
            # from scipy import sparse
            # import numpy as np
            # X_csr = sparse.csr_matrix(tt_adj.cpu().numpy())
            # print(X_csr)

            # sent_token
            st_adj = np.zeros((topic_label.shape[0], input_is.shape[0]))
            start = 0
            span_count = 0
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    st_adj[span_count][start:ends[i][j] - starts[i][j] + 1 +start] = 1
                    start = start + (ends[i][j] - starts[i][j] + 1)
                    span_count+=1
                # print(st_adj[i])

            # token_sent
            ts_adj = st_adj.T

            st_adj = torch.Tensor(st_adj).float().to(self.device)
            ts_adj = torch.Tensor(ts_adj).float().to(self.device)

            # sent_sent
            topic_label = topic_label.contiguous().view(-1, 1)
            ss_adj = torch.eq(topic_label, topic_label.T).float().to(self.device)  # 逐元素的比较，若相同位置的两个元素相同，则返回True

            # adj
            adj = np.zeros((topic_label.shape[0] + input_is.shape[0], topic_label.shape[0] + input_is.shape[0]))
            adj = torch.Tensor(adj).float().to(self.device)
            adj[:topic_label.shape[0], :topic_label.shape[0]] = ss_adj
            adj[:topic_label.shape[0], topic_label.shape[0]:] = st_adj
            adj[topic_label.shape[0]:, :topic_label.shape[0]] = ts_adj
            adj[topic_label.shape[0]:, topic_label.shape[0]:] = tt_adj

            aaa = sum(sum(adj - adj.T))
            a = adj.cpu().numpy()
            gcn_out = self.gat(span_embs, adj)

            stance_out = gcn_out[:topic_label.shape[0], :]

            logits = self.stance_classifier(stance_out)
            logits = F.softmax(logits, dim=-1)
            if label_stance is not None:  # training
                label_ = []
                for i in label_stance:
                    for j in i:
                        label_.append(j)
                label_ = torch.Tensor(label_).to(self.device).view(-1, 1)
                label_ = label_.to(torch.int64)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, 2),
                    label_.view(-1)
                )
                return loss, logits
            else: # inference
                return torch.argmax(logits, dim=-1), logits


        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            # span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            # b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn = self.span_softmax(attn)
            attn = attn * span2term

            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn, values)

            #output3 = output + output2
            #output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output2)
            logits_AS = F.softmax(logits_AS, dim=-1)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':
            logits_AM2 = torch.Tensor(logits_AM2).to(self.device)

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class TokenBERT_1122_nstance(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_1122_nstance, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        #self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 968, 968)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(300, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=2)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer= 1)
        self.stance_classifier = nn.Linear(300, 2)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None, starts=None, ends=None, topic_indices=None, label_stance=None, token_spacy_ids= None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        #得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_emb = torch.cat((word_reps, pos_emb), 2)
        word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        #LSTM
        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #syn_lstm
        feature_out = self.syn_lstm(word_emb, sent_length, graph)
        #word_rep = self.word_drop(feature_out)

        #star_transformer
        x = self.word2star(feature_out)
        h = self.posi(x)
        s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        for idx in range(self.HP_star_layer):
            h, s = self.star_transformer(h, x, s)

        output = h

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)
            logits_AM = F.softmax(logits_AM, dim=-1)
            logits_AS = F.softmax(logits_AS, dim=-1)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        # if model == 'stance':
        #
        #     #last_hidden_state_cls = torch.concat((last_hidden_state_cls, token), dim=0)
        #     topic_ = self.embed(topic_indices)
        #     topic_label = []
        #     span_embs = None
        #     for i in range(len(starts)):
        #         for j in range(len(starts[i])):
        #             hi=output[i][starts[i][j]]
        #             hj=output[i][ends[i][j]]
        #             topici=topic_[i]
        #             if span_embs == None:
        #                 span_embs=(hi+hj+topici).view(1, -1)
        #             else:
        #                 span_embs = torch.concat((span_embs, (hi+hj+topici).view(1, -1)), dim=0)
        #             topic_label.append(topic_indices[i])
        #
        #     topic_label = torch.Tensor(topic_label).float().to(self.device).view(-1, 1)
        #
        #     #token_spacy_ids = []
        #     token_embs = []
        #     input_is = []
        #     # token_token
        #     for i in range(len(starts)):
        #         for j in range(len(starts[i])):
        #             input_i = token_spacy_ids[i][starts[i][j]:ends[i][j]+1]
        #             token = output[i][starts[i][j]:ends[i][j]+1, :]
        #             input_is.extend(input_i)
        #             token_embs.append(token)
        #             span_embs = torch.concat((span_embs, token), dim=0)
        #
        #     input_is = torch.Tensor(input_is).float().to(self.device).view(-1, 1)
        #     # labels = input_is.contiguous().view(-1, 1)
        #     tt_adj = torch.eq(input_is, input_is.T).float().to(self.device)
        #     # from scipy import sparse
        #     # import numpy as np
        #     # X_csr = sparse.csr_matrix(tt_adj.cpu().numpy())
        #     # print(X_csr)
        #
        #     # sent_token
        #     st_adj = np.zeros((topic_label.shape[0], input_is.shape[0]))
        #     start = 0
        #     span_count = 0
        #     for i in range(len(starts)):
        #         for j in range(len(starts[i])):
        #             st_adj[span_count][start:ends[i][j] - starts[i][j] + 1 +start] = 1
        #             start = start + (ends[i][j] - starts[i][j] + 1)
        #             span_count+=1
        #         # print(st_adj[i])
        #
        #     # token_sent
        #     ts_adj = st_adj.T
        #
        #     st_adj = torch.Tensor(st_adj).float().to(self.device)
        #     ts_adj = torch.Tensor(ts_adj).float().to(self.device)
        #
        #     # sent_sent
        #     topic_label = topic_label.contiguous().view(-1, 1)
        #     ss_adj = torch.eq(topic_label, topic_label.T).float().to(self.device)  # 逐元素的比较，若相同位置的两个元素相同，则返回True
        #
        #     # adj
        #     adj = np.zeros((topic_label.shape[0] + input_is.shape[0], topic_label.shape[0] + input_is.shape[0]))
        #     adj = torch.Tensor(adj).float().to(self.device)
        #     adj[:topic_label.shape[0], :topic_label.shape[0]] = ss_adj
        #     adj[:topic_label.shape[0], topic_label.shape[0]:] = st_adj
        #     adj[topic_label.shape[0]:, :topic_label.shape[0]] = ts_adj
        #     adj[topic_label.shape[0]:, topic_label.shape[0]:] = tt_adj
        #
        #     aaa = sum(sum(adj - adj.T))
        #     a = adj.cpu().numpy()
        #     gcn_out = self.gat(span_embs, adj)
        #
        #     stance_out = gcn_out[:topic_label.shape[0], :]
        #
        #     logits = self.stance_classifier(stance_out)
        #     logits = F.softmax(logits, dim=-1)
        #     if label_stance is not None:  # training
        #         label_ = []
        #         for i in label_stance:
        #             for j in i:
        #                 label_.append(j)
        #         label_ = torch.Tensor(label_).to(self.device).view(-1, 1)
        #         label_ = label_.to(torch.int64)
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(
        #             logits.view(-1, 2),
        #             label_.view(-1)
        #         )
        #         return loss, logits
        #     else: # inference
        #         return torch.argmax(logits, dim=-1), logits


        if model == 'span2term':
            span2term = span2term.unsqueeze(2)
            a = span2term.cpu().numpy()
            # span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
            # b = span2term.cpu().numpy()
            # output_mask = output * term2span
            #queries = output
            queries = self.W_q(output)
            #keys = output
            keys = self.W_k(output)
            #values = output
            values = self.W_v(output)

            B, Nt, E = queries.shape
            queries = queries / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(queries, keys.transpose(1, 2))
            attn = self.span_softmax(attn)
            attn = attn * span2term

            c = attn.detach().cpu().numpy()
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output2 = torch.bmm(attn, values)

            #output3 = output + output2
            #output3 = self.norm1(output + output2)
            logits_AS = self.fc_linear_as(output2)
            logits_AS = F.softmax(logits_AS, dim=-1)

            if AS_labels is not None:  # training
                return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
            else:  # inference
                return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':

            logits_AM_f = logits_AM1
            logits_AS_f = logits_AS1 + logits_AS2


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class TokenBERT_1122_nIM(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_1122_nIM, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        #self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        #self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 968, 968)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(300, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=2)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8,300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer= 1)
        self.stance_classifier = nn.Linear(300, 2)

    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None, starts=None, ends=None, topic_indices=None, label_stance=None, token_spacy_ids= None):
        #embedding
        # char
        char_features = self.char_feature.get_last_hiddens(char_ids, char_len)
        # pos
        pos_emb = self.pos_label_embedding(pos_ids)



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        #得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_emb = torch.cat((word_reps, pos_emb), 2)
        word_emb = torch.cat((word_emb, char_features), 2)

        #word_reps = self.dropout(word_reps)
        #LSTM
        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        #syn_lstm
        feature_out = self.syn_lstm(word_emb, sent_length, graph)
        #word_rep = self.word_drop(feature_out)

        #star_transformer
        x = self.word2star(feature_out)
        h = self.posi(x)
        s = torch.mean(x, 1)  # s是e各行/列的算数平均值
        for idx in range(self.HP_star_layer):
            h, s = self.star_transformer(h, x, s)

        output = h

        if model == 'embedding':

            logits_AM = self.fc_linear1(output)
            logits_AS = self.fc_linear2(output)
            logits_AM = F.softmax(logits_AM, dim=-1)
            logits_AS = F.softmax(logits_AS, dim=-1)


            if AM_labels is not None:  # training
                return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte()), logits_AM, logits_AS
            else:  # inference
                return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte()), logits_AM, logits_AS

        if model == 'stance':

            #last_hidden_state_cls = torch.concat((last_hidden_state_cls, token), dim=0)
            topic_ = self.embed(topic_indices)
            topic_label = []
            span_embs = None
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    hi=output[i][starts[i][j]]
                    hj=output[i][ends[i][j]]
                    topici=topic_[i]
                    if span_embs == None:
                        span_embs=(hi+hj+topici).view(1, -1)
                    else:
                        span_embs = torch.concat((span_embs, (hi+hj+topici).view(1, -1)), dim=0)
                    topic_label.append(topic_indices[i])

            topic_label = torch.Tensor(topic_label).float().to(self.device).view(-1, 1)

            #token_spacy_ids = []
            token_embs = []
            input_is = []
            # token_token
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    input_i = token_spacy_ids[i][starts[i][j]:ends[i][j]+1]
                    token = output[i][starts[i][j]:ends[i][j]+1, :]
                    input_is.extend(input_i)
                    token_embs.append(token)
                    span_embs = torch.concat((span_embs, token), dim=0)

            input_is = torch.Tensor(input_is).float().to(self.device).view(-1, 1)
            # labels = input_is.contiguous().view(-1, 1)
            tt_adj = torch.eq(input_is, input_is.T).float().to(self.device)
            # from scipy import sparse
            # import numpy as np
            # X_csr = sparse.csr_matrix(tt_adj.cpu().numpy())
            # print(X_csr)

            # sent_token
            st_adj = np.zeros((topic_label.shape[0], input_is.shape[0]))
            start = 0
            span_count = 0
            for i in range(len(starts)):
                for j in range(len(starts[i])):
                    st_adj[span_count][start:ends[i][j] - starts[i][j] + 1 +start] = 1
                    start = start + (ends[i][j] - starts[i][j] + 1)
                    span_count+=1
                # print(st_adj[i])

            # token_sent
            ts_adj = st_adj.T

            st_adj = torch.Tensor(st_adj).float().to(self.device)
            ts_adj = torch.Tensor(ts_adj).float().to(self.device)

            # sent_sent
            topic_label = topic_label.contiguous().view(-1, 1)
            ss_adj = torch.eq(topic_label, topic_label.T).float().to(self.device)  # 逐元素的比较，若相同位置的两个元素相同，则返回True

            # adj
            adj = np.zeros((topic_label.shape[0] + input_is.shape[0], topic_label.shape[0] + input_is.shape[0]))
            adj = torch.Tensor(adj).float().to(self.device)
            adj[:topic_label.shape[0], :topic_label.shape[0]] = ss_adj
            adj[:topic_label.shape[0], topic_label.shape[0]:] = st_adj
            adj[topic_label.shape[0]:, :topic_label.shape[0]] = ts_adj
            adj[topic_label.shape[0]:, topic_label.shape[0]:] = tt_adj

            aaa = sum(sum(adj - adj.T))
            a = adj.cpu().numpy()
            gcn_out = self.gat(span_embs, adj)

            stance_out = gcn_out[:topic_label.shape[0], :]

            logits = self.stance_classifier(stance_out)
            logits = F.softmax(logits, dim=-1)
            if label_stance is not None:  # training
                label_ = []
                for i in label_stance:
                    for j in i:
                        label_.append(j)
                label_ = torch.Tensor(label_).to(self.device).view(-1, 1)
                label_ = label_.to(torch.int64)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, 2),
                    label_.view(-1)
                )
                return loss, logits
            else: # inference
                return torch.argmax(logits, dim=-1), logits


        # if model == 'span2term':
        #     span2term = span2term.unsqueeze(2)
        #     a = span2term.cpu().numpy()
        #     # span2term = span2term.expand(output.shape[0], output.shape[1], output.shape[1])
        #     # b = span2term.cpu().numpy()
        #     # output_mask = output * term2span
        #     #queries = output
        #     queries = self.W_q(output)
        #     #keys = output
        #     keys = self.W_k(output)
        #     #values = output
        #     values = self.W_v(output)
        #
        #     B, Nt, E = queries.shape
        #     queries = queries / math.sqrt(E)
        #     # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        #     attn = torch.bmm(queries, keys.transpose(1, 2))
        #     attn = self.span_softmax(attn)
        #     attn = attn * span2term
        #
        #     c = attn.detach().cpu().numpy()
        #     # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        #     output2 = torch.bmm(attn, values)
        #
        #     #output3 = output + output2
        #     #output3 = self.norm1(output + output2)
        #     logits_AS = self.fc_linear_as(output2)
        #     logits_AS = F.softmax(logits_AS, dim=-1)
        #
        #     if AS_labels is not None:  # training
        #         return -self.crf_as(logits_AS, AS_labels, label_mask.byte()), logits_AS
        #     else:  # inference
        #         return self.crf_as.decode(logits_AS, label_mask.byte()), logits_AS

        if model == 'final':
            logits_AM2 = torch.Tensor(logits_AM2).to(self.device)

            logits_AM_f = logits_AM1 + logits_AM2
            logits_AS_f = logits_AS1


            if AM_labels is not None:  # training
                return -self.crf_am_f(logits_AM_f, AM_labels, label_mask.byte()) - self.crf_as_f(logits_AS_f, AS_labels, label_mask.byte())
            else:  # inference
                return self.crf_am_f.decode(logits_AM_f, label_mask.byte()), self.crf_as_f.decode(logits_AS_f, label_mask.byte())

class TokenBERT_LSTM_CRF_our(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_LSTM_CRF_our, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        # self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        # self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 768, 768)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(300, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=2)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8, 300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer=1)
        self.stance_classifier = nn.Linear(300, 2)


    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, pieces2word, graph, label_mask, batch_head=None, batch_tail=None, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        # 得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        # syn_lstm
        feature_out = self.syn_lstm(word_reps, sent_length, graph)
        # word_rep = self.word_drop(feature_out)

        # feature_out1 = self.dropout(feature_out)
        logits_AM = self.fc_linear1(feature_out)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte())

        # if AM_labels is not None:  # training
        #     return -self.crf1(logits_AM, AM_labels, label_mask.byte(), reduction= 'token_mean') - self.crf2(logits_AS, AS_labels, label_mask.byte(), reduction= 'token_mean')
        # else:  # inference
        #     return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())


class TokenBERT_LSTM_CRF_bert(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_LSTM_CRF_bert, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder1 = nn.LSTM(768, 150, num_layers=1, batch_first=True, bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        #self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)

        self.drophead = nn.Dropout(0.5)

        self.crf1 = CRF(7, batch_first=self.batch_first)


    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, pieces2word, graph, label_mask, batch_head=None, batch_tail=None, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        # 得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        # syn_lstm
        #feature_out = self.syn_lstm(word_emb, sent_length, graph)
        feature_out, (hidden, _) = self.encoder1(word_reps)
        # word_rep = self.word_drop(feature_out)

        # feature_out1 = self.dropout(feature_out)
        logits_AM = self.fc_linear1(feature_out)

        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte())

        # if AM_labels is not None:  # training
        #     return -self.crf1(logits_AM, AM_labels, label_mask.byte(), reduction= 'token_mean') - self.crf2(logits_AS, AS_labels, label_mask.byte(), reduction= 'token_mean')
        # else:  # inference
        #     return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())


class TokenBERT_abam_LSTM_CRF_our(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_abam_LSTM_CRF_our, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder = nn.LSTM(768, 150, num_layers=1, batch_first=True,
                               bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        # self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.fc_linear_am = torch.nn.Linear(300, 7)
        self.fc_linear_as = torch.nn.Linear(300, 4)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)

        self.crf_am = CRF(7, batch_first=self.batch_first)
        self.crf_as = CRF(4, batch_first=self.batch_first)

        self.crf_am_f = CRF(7, batch_first=self.batch_first)
        self.crf_as_f = CRF(4, batch_first=self.batch_first)

        self.char_feature = CharBiLSTM(device)
        # self.gat = GraphAttentionLayer(bert_config.hidden_size, 2*opt.hidden_dim, dropout=0.0, alpha=0.2, concat=True)

        self.syn_lstm = GAT(device, 300, 768, 768)  ### lstm hidden size
        self.pos_label_embedding = nn.Embedding(18, 100).to(device)
        self.word_drop = nn.Dropout(0.3).to(device)

        self.HP_star_glu = 2
        self.HP_star_dropout = 0.1
        self.HP_star_head = 5
        self.HP_star_layer = 6

        self.word2star = nn.Linear(300, 300)
        self.posi = PositionalEncoding(300, 0.5)
        self.star_transformer = StarEncoderLayer(
            d_model=300,
            n_head=self.HP_star_head,
            d_k=300,
            d_v=300,
            glu_type=self.HP_star_glu,
            dropout=self.HP_star_dropout
        )

        self.W_q = nn.Linear(300, 300, bias=True)
        self.W_k = nn.Linear(300, 300, bias=True)
        self.W_v = nn.Linear(300, 300, bias=True)

        self.term_softmax = nn.Softmax(dim=1)
        self.span_softmax = nn.Softmax(dim=2)

        self.norm1 = LayerNorm(300)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(np.random.rand(8, 300).astype(np.float32)))
        self.gat = GAT2(300, 300, 0, 0.5, alpha=0.1, nheads=3, layer=1)
        self.stance_classifier = nn.Linear(300, 2)


    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, batch_head=None, batch_tail=None, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        # 得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        # syn_lstm
        feature_out = self.syn_lstm(word_reps, sent_length, graph)
        # word_rep = self.word_drop(feature_out)

        # feature_out1 = self.dropout(feature_out)
        logits_AM = self.fc_linear1(feature_out)

        logits_AS = self.fc_linear2(feature_out)


        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte())-self.crf2(logits_AS, AS_labels, label_mask.byte())
        else:  # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())

        # if AM_labels is not None:  # training
        #     return -self.crf1(logits_AM, AM_labels, label_mask.byte(), reduction= 'token_mean') - self.crf2(logits_AS, AS_labels, label_mask.byte(), reduction= 'token_mean')
        # else:  # inference
        #     return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())

class TokenBERT_abam_LSTM_CRF_bert(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True, device=None):
        super(TokenBERT_abam_LSTM_CRF_bert, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.encoder1 = nn.LSTM(768, 150, num_layers=1, batch_first=True, bidirectional=True)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        #self.fc_linear1 = torch.nn.Linear(768, 5)
        self.device = device
        self.fc_linear1 = torch.nn.Linear(300, 7)
        self.fc_linear2 = torch.nn.Linear(300, 4)

        self.drophead = nn.Dropout(0.5)

        self.crf1 = CRF(7, batch_first=self.batch_first)
        self.crf2 = CRF(4, batch_first=self.batch_first)


    def forward(self, model, input_ids, attention_mask, token_type_ids, sent_length, char_ids, char_len, pos_ids, pieces2word, graph, label_mask, batch_head=None, batch_tail=None, term2span=None, span2term=None, AM_labels=None, AS_labels=None, logits_AM1=None, logits_AS1=None, logits_AM2=None, logits_AS2=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_embs = outputs[0]

        # 得到bert转spacy的表示
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        # Max pooling word representations from pieces
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        # packed_outs, (hidden, _) = self.encoder(packed_embs)
        # word_emb, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=85)

        # syn_lstm
        #feature_out = self.syn_lstm(word_emb, sent_length, graph)
        feature_out, (hidden, _) = self.encoder1(word_reps)
        # word_rep = self.word_drop(feature_out)

        # feature_out1 = self.dropout(feature_out)
        logits_AM = self.fc_linear1(feature_out)
        logits_AS = self.fc_linear2(feature_out)

        if AM_labels is not None:  # training
            return -self.crf1(logits_AM, AM_labels, label_mask.byte()) - self.crf2(logits_AS, AS_labels, label_mask.byte())
        else: # inference
            return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())

        # if AM_labels is not None:  # training
        #     return -self.crf1(logits_AM, AM_labels, label_mask.byte(), reduction= 'token_mean') - self.crf2(logits_AS, AS_labels, label_mask.byte(), reduction= 'token_mean')
        # else:  # inference
        #     return self.crf1.decode(logits_AM, label_mask.byte()), self.crf2.decode(logits_AS, label_mask.byte())
