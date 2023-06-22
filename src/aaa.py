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
