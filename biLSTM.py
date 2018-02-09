# -*- coding: utf-8 -*-
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mask import SimpleMask

use_cuda = torch.cuda.is_available()

word_to_ix = {UNK:0}
pos1_to_ix = {UNK:0}
pos2_to_ix = {UNK:0}
tag_to_ix = {UNK:0}
ix_to_tag = {UNK}

class EncoderRNN(nn.Module):
    def __init__(self, word_size, word_dim, pretrain_size, pretrain_dim, pretrain_embeddings, pos_size, pos_dim, input_dim, hidden_dim, n_layers=1, dropout_p=0.0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(word_size, word_dim)
        self.pretrain_embeds = nn.Embedding(pretrain_size, pretrain_dim)
        self.pretrain_embeds.weight = nn.Parameter(pretrain_embeddings, False)
        self.pos_embeds = nn.Embedding(pos_size, pos_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        self.embeds2input = nn.Linear(word_dim + pretrain_dim + pos_dim, input_dim)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.n_layers, bidirectional=True)

    def forward(self, sentence, hidden, train=True):
        word_embedded = self.word_embeds(sentence[0])
        pretrain_embedded = self.pretrain_embeds(sentence[1])
        pos_embedded = self.pos_embeds(sentence[2])

        if train:
            word_embedded = self.dropout(word_embedded)
            pos_embedded = self.dropout(pos_embedded)
            self.lstm.dropout = self.dropout_p

        embeds = self.tanh(self.embeds2input(torch.cat((word_embedded, pretrain_embedded, pos_embedded), 1))).view(len(sentence[0]),1,-1)
        output, hidden = self.lstm(embeds, hidden)
        return output, hidden

    def initHidden(self):
        if use_cuda:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda())
            return result
        else:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)))
            return result


def train(sentence_variable, gold_variable, bilstm, bilstm_optimizer, criterion, back_prop=True):
    bilstm_hidden = bilstm.initHidden()

    bilstm_optimizer.zero_grad()
   
    bilstm_output = bilstm(sentence_variable, bilstm_hidden)

    loss = 0


    while idx < sentence_length:

        decoder_input = Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[decoder.tags_info.SOS]]))
        if back_prop== False:
            decoder_input.volatile = True
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_input = torch.cat((decoder_input, target_variable[idx]))

        decoder_hidden = (encoder_output[idx].unsqueeze(0), decoder.initC())
        
        decoder_output = decoder(decoder_input, decoder_hidden, encoder_output, train=True, mask_variable=mask_variable[idx])

        index_variable = torch.cat((gold_variable[idx], Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[decoder.tags_info.EOS]]))))

        index_variable = index_variable.cuda() if use_cuda else index_variable
        #print mask_variable[idx]
        #print decoder_output
        #print gold_variable
        #exit(1)
        tmp_loss = criterion(decoder_output, index_variable)

        loss += tmp_loss.data[0] / decoder_input.size(0)
        
        idx += 1

    if back_prop == False:
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
    
    return loss / sentence_length 

def decode(sentence_variable, target_variable, encoder, decoder):
    encoder_hidden = encoder.initHidden()
    sentence_length = sentence_variable.size(0)
    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)

    idx = 0
    tokens = []
    while idx < sentence_length:
        decoder_input = Variable(torch.LongTensor([decoder.tags_info.tag_to_ix[SOS]]), volatile=True)
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_hidden = (encoder_output[idx].unsqueeze(0), decoder.initC(back_prop=False))
        tokens.append(decoder(decoder_input, decoder_hidden, encoder_output, train=False))
        idx += 1
    return tokens

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(trn_instances, dev_instances, bilstm, print_every=100, evaluate_every=1000, learning_rate=0.001):
    print_loss_total = 0  # Reset every print_every

    bilstm_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate, weight_decay=1e-4)

    criterion = nn.NLLLoss()

    idx = -1
    iter = 0
    while True:
        idx += 1
        iter += 1
        if idx == len(trn_instances):
            idx = 0

        sentence_variable = []
        gold_variable = []

        if use_cuda:
            sentence_variable.append(Variable(trn_instances[idx][0]).cuda())
            sentence_variable.append(Variable(trn_instances[idx][1]).cuda())
            sentence_variable.append(Variable(trn_instances[idx][2]).cuda())
            gold_variable.append(Variable(trn_instances[idx][-1]).cuda())
        else:
            sentence_variable.append(Variable(trn_instances[idx][0]))
            sentence_variable.append(Variable(trn_instances[idx][1]))
            sentence_variable.append(Variable(trn_instances[idx][2]))
            gold_variable.append(Variable(trn_instances[idx][-1]))

        loss = train(sentence_variable, gold_variable, bilstm, bilstm_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch %.6f : %.10f' % (iter*1.0 / len(trn_instances), print_loss_avg))

        if iter % evaluate_every == 0:
            evaluate(dev_instances, bilstm, str(int(iter/evaluate_every)))

def evaluate(instances, encoder, decoder, part):
    out = open("dev_output/"+part,"w")
    for instance in instances:
        sentence_variable = []
        target_variable = Variable(torch.LongTensor([ x[1] for x in instance[-1]]))
        if use_cuda:
            sentence_variable.append(Variable(instance[0]).cuda())
            sentence_variable.append(Variable(instance[1]).cuda())
            sentence_variable.append(Variable(instance[2]).cuda())
            target_variable = target_variable.cuda()
        else:
            sentence_variable.append(Variable(instance[0]))
            sentence_variable.append(Variable(instance[1]))
            sentence_variable.append(Variable(instance[2]))
        tokens = decode(sentence_variable, target_variable, encoder, decoder)

	output = []
        for type, tok in tokens:
            if type >= 0:
                output.append(decoder.tags_info.ix_to_lemma[tok])
            else:
                output.append(decoder.tags_info.ix_to_tag[tok])
        out.write(" ".join(output)+"\n")
	out.flush()
    out.close()
#####################################################################################
#####################################################################################
#####################################################################################
# main

from utils import readfile2
from utils import readpretrain
from utils import data2instance2

trn_file = "train.actions"
dev_file = "dev.actions"
tst_file = "test.actions"
pretrain_file = "sskip.100.vectors"
tag_info_file = "tag.info"
trn_file = "train.actions.part"
dev_file = "dev.actions.part"
tst_file = "test.actions.part"
pretrain_file = "sskip.100.vectors.part"
UNK = "<UNK>"

trn_data = readfile2(trn_file)
for sentence, _, postags1, postags2, tags in trn_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for postag in postags1:
        if postag not in pos1_to_ix:
            pos1_to_ix[postag] = len(pos1_to_ix)
    for postag in postags2:
        if postag not in pos2_to_ix:
            pos2_to_ix[postag] = len(pos2_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
            ix_to_tag.append(tag)

pretrain_to_ix = {UNK:0}
pretrain_embeddings = [ [0. for i in range(100)] ] # for UNK 
pretrain_data = readpretrain(pretrain_file)
for one in pretrain_data:
    pretrain_to_ix[one[0]] = len(pretrain_to_ix)
    pretrain_embeddings.append([float(a) for a in one[1:]])
print "pretrain dict size:", len(pretrain_to_ix)

dev_data = readfile(dev_file)
tst_data = readfile(tst_file)

print "word dict size: ", len(word_to_ix)
print "pos1 dict size: ", len(pos1_to_ix)
print "pos2 dict size: ", len(pos2_to_ix)
print "global tag dict size: ", tags_info.tag_size

WORD_EMBEDDING_DIM = 128
PRETRAIN_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 128

INPUT_DIM = 256
ENCODER_HIDDEN_DIM = 512

encoder = EncoderRNN(len(word_to_ix), WORD_EMBEDDING_DIM, len(pretrain_to_ix), PRETRAIN_EMBEDDING_DIM, torch.FloatTensor(pretrain_embeddings), len(pos1_to_ix), POS_EMBEDDING_DIM, INPUT_DIM, ENCODER_HIDDEN_DIM, n_layers=2, dropout_p=0.1)

###########################################################
# prepare training instance
trn_instances = data2instance2(trn_data, [(word_to_ix,0), (pretrain_to_ix,0), (pos1_to_ix,0), (pos2_to_ix,0), (tag_to_ix,0)])
print "trn size: " + str(len(trn_instances))
###########################################################
# prepare development instance
dev_instances = data2instance2(dev_data, [(word_to_ix,0), (pretrain_to_ix,0), (pos1_to_ix,0), (pos2_to_ix,0), (tag_to_ix,0)])
print "dev size: " + str(len(dev_instances))
###########################################################
# prepare test instance
tst_instances = data2instance2(tst_data, [(word_to_ix,0), (pretrain_to_ix,0), (pos1_to_ix,0), (pos2_to_ix,0), (tag_to_ix,0)])
print "tst size: " + str(len(tst_instances))

print "GPU", use_cuda
if use_cuda:
    encoder = encoder.cuda()

trainIters(trn_instances, dev_instances, encoder, print_every=1000, evaluate_every=50000, learning_rate=0.0005)

