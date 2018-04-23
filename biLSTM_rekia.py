# -*- coding: utf-8 -*-
##
#   200 glove pretrained
#   adding windows
#   re-init parameter
##

import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import math
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(12345678)

torch.manual_seed(12345678)

dev_out_dir = sys.argv[1]+"_dev/"
tst_out_dir = sys.argv[1]+"_tst/"
model_dir = sys.argv[1]+"_model/"

UNK = "<UNK>"
word_to_ix = {UNK:0}
ix_to_word = [UNK]
cap_to_ix = {"UCAP":0, "CAP":1}
ix_to_cap = ["UCAP", "CAP"]
suffix_to_ix = {UNK:0}
ix_to_suffix = [UNK]
tag_to_ix = {}
ix_to_tag = []


WORD_EMBEDDING_DIM = 300
CAP_EMBEDDING_DIM = 5
SUFFIX_EMBEDDIN_DIM = 5


INPUT_DIM = WORD_EMBEDDING_DIM + CAP_EMBEDDING_DIM + SUFFIX_EMBEDDIN_DIM
HIDDEN_DIM = 400
DROPOUT_P = 0.5
LEARNING_RATE = 0.001

class EncoderRNN(nn.Module):
    def __init__(self, word_size, word_dim, cap_size, cap_dim, suffix_size, suffix_dim, input_dim, hidden_dim, n_layers=2, dropout_p=0.5):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(word_size, word_dim)
        self.cap_embeds = nn.Embedding(cap_size, cap_dim)
        self.suffix_embeds = nn.Embedding(suffix_size, suffix_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.n_layers, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                tmp = torch.nn.init.orthogonal(param)

        self.out = self.linear_init(nn.Linear(hidden_dim, tag_size))

    def forward(self, sentence, hidden, train=True):
        word_embedded = self.word_embeds(sentence[0])
        cap_embedded = self.cap_embeds(sentence[1])
        suffix_embedded = self.suffix_embeds(sentence[2])

        if train:
            word_embedded = self.dropout(word_embedded)
            cap_embedded = self.dropout(cap_embedded)
            suffix_embedded = self.dropout(suffix_embedded)
            self.lstm.dropout = self.dropout_p

        embeds = torch.cat((word_embedded, pretrain_embedded, pos_embedded), 1).view(sentence[0])
        output, hidden = self.lstm(embeds, hidden)
        return self.out(output)

    def linear_init(self, linear):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(linear.weight)
        linear.weight = torch.nn.Parameter((torch.nn.init.normal(linear.weight, 0, 1.0 / math.sqrt(fan_in))*0.1).data)
        linear.bias = torch.nn.Parameter((torch.nn.init.normal(linear.bias, 0, 1)*0.1).data)
        return linear
	
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
    sentence_length = sentence_variable[0].size(0)
    bilstm_output = bilstm(sentence_variable, bilstm_hidden)
    dist = F.log_softmax(bilstm_output,1)
    loss = criterion(dist, gold_variable)

    if back_prop == True:
        bilstm_optimizer.zero_grad()
        loss.backward()
        bilstm_optimizer.step()
    
    return loss / sentence_length 

def decode(sentence_variable, bilstm):
    bilstm_hidden = bilstm.initHidden()
    bilstm_output = bilstm(sentence_variable, bilstm_hidden)
    scores, indexs = torch.max(bilstm_output, 1)
    return indexs.view(-1).data.tolist()

def trainIters(trn_instances, dev_instances, tst_instances, bilstm, print_every=100, evaluate_every=1000, learning_rate=0.001):
    print_loss_total = 0  # Reset every print_every

    #bilstm_optimizer = optim.Adam(filter(lambda p: p.requires_grad, bilstm.parameters()), lr=learning_rate)
    #bilstm_optimizer = optim.SGD(filter(lambda p: p.requires_grad, bilstm.parameters()), lr=learning_rate)
    bilstm_optimizer = optim.Adam(bilstm.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    idx = -1
    iter = 0
    while True:
        idx += 1
        iter += 1
        if idx == len(trn_instances):
            idx = 0

        input_words = []
        for i in range(len(trn_instances[idx][0])):
            w = trn_instances[idx][0][i]
            if w in rare_word_ix and random.uniform(0,1) <= 0.1:
                input_words.append(trn_instances[idx][1][i])
            else:
                input_words.append(w)

        sentence_variable = []
        gold_variable = []
        if use_cuda:
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][0])).cuda())
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][1])).cuda())
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][2])).cuda())
            gold_variable = Variable(torch.LongTensor(trn_instances[idx][-1])).cuda()
        else:
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][0])))
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][1])))
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][2])))
            gold_variable = Variable(torch.LongTensor(trn_instances[idx][-1]))

        loss = train(sentence_variable, gold_variable, bilstm, bilstm_optimizer, criterion, True)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch %.6f : %.10f' % (iter*1.0 / len(trn_instances), print_loss_avg))

        if iter % evaluate_every == 0:
            dev_acc = evaluate(dev_instances, bilstm, dev_out_dir+str(int(iter/evaluate_every))+".pred")
            tst_acc = evaluate(tst_instances, bilstm, tst_out_dir+str(int(iter/evaluate_every))+".pred")
            print('dev %d, dev_acc: %.10f, tst_acc: %.10f' %(int(iter/evaluate_every), dev_acc, tst_acc))
def evaluate(instances, bilstm, dir_path):
    out = open(dir_path,"w")
    correct = 0
    total = 0
    for instance in instances:
        sentence_variable = []
        if use_cuda:
            sentence_variable.append(Variable(torch.LongTensor(instance[0]), volatile=True).cuda())
            sentence_variable.append(Variable(torch.LongTensor(instance[1]), volatile=True).cuda())
            sentence_variable.append(Variable(torch.LongTensor(instance[2]), volatile=True).cuda())
        else:
            sentence_variable.append(Variable(torch.LongTensor(instance[0]), volatile=True))
            sentence_variable.append(Variable(torch.LongTensor(instance[1]), volatile=True))
            sentence_variable.append(Variable(torch.LongTensor(instance[2]), volatile=True))
        indexs = decode(sentence_variable, bilstm)

        assert len(indexs) == len(instance[-1])
        i = 0
        for idx in indexs:
            out.write(ix_to_tag[idx]+"\n")
            if idx == instance[-1][i]:
                correct += 1
            i += 1
    total += len(indexs)
    out.write("\n")
    out.flush()
    out.close()
    return correct * 1.0 /total
#####################################################################################
#####################################################################################
#####################################################################################
# main

from utils import readfile_rekia
from utils import readpretrain
from utils import data2instance3
#from utils import all_possible_UNK

trn_file = "train.input"
dev_file = "dev.input"
tst_file = "test.input"
pretrain_file = "sskip.100.vectors"
pretrain_file = "glove.6B.300d.txt"
#trn_file = "train.input.part"
#dev_file = "train.input.part"
#tst_file = "test.actions.part"
#pretrain_file = "sskip.100.vectors.part"

pretrain_embeddings = [ [0. for i in range(WORD_EMBEDDING_DIM)] ] # for UNK
unk_embeddings = [ 0. for i in range(WORD_EMBEDDING_DIM)]
pretrain_data = readpretrain(pretrain_file)
for one in pretrain_data:
    word_to_ix[one[0]] = len(word_to_ix)
    ix_to_word.append(one[0])
    pretrain_embeddings.append([float(a) for a in one[1:]])
    for i in range(WORD_EMBEDDING_DIM):
        unk_embeddings[i] += float(one[i+1])
for i in range(WORD_EMBEDDING_DIM):
    unk_embeddings[i] /= len(pretrain_embeddings) - 1
pretrain_embeddings[0] = unk_embeddings
print "pretrain dict size:", len(pretrain_to_ix)

trn_data = readfile3(trn_file)
for _, _, suffixs, tags in trn_data:
    for suffix in suffixs:
        if suffix not in suffix_to_ix:
            suffix_to_ix[suffix] = len(suffix_to_ix)
            ix_to_suffix.append(suffix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
            ix_to_tag.append(tag)

dev_data = readfile3(dev_file)
for _, _, _, tags in dev_data:
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
            ix_to_tag.append(tag)
tst_data = readfile3(tst_file)
for _, _, _, tags in tst_data:
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
            ix_to_tag.append(tag)

print "word dict size: ", len(word_to_ix)
print "cap dict size: ", len(cap_to_ix)
print "suffix dict size: ", len(suffix_to_ix)
print "tag dict size: ", len(tag_to_ix)

bilstm = EncoderRNN(len(word_to_ix), WORD_EMBEDDING_DIM, torch.FloatTensor(pretrain_embeddings), len(cap_to_ix), CAP_EMBEDDING_DIM, len(suffix_to_ix), SUFFIX_EMBEDDING_DIM, INPUT_DIM, HIDDEN_DIM, n_layers=2, dropout_p=DROPOUT_P)

###########################################################
# prepare training instance
trn_instances, unks, total_ws = data2instance2(trn_data, [(word_to_ix,0), (cap_to_ix,-1), (suffix_to_ix,0), (tag_to_ix,-1)])
print "trn size: " + str(len(trn_instances))
print "word unk:",
print unks[0],
print total_ws[0],
print unks[0]*1.0 / total_ws[0]
print "cap unk:",
print unks[1],
print total_ws[1],
print unks[1]*1.0 / total_ws[1]
print "suffix unk:",
print unks[2],
print total_ws[2],
print unks[2]*1.0 / total_ws[2]
print "tag unk:",
print unks[3],
print total_ws[3],
print unks[3]*1.0 / total_ws[3]
###########################################################
# prepare development instance
dev_instances, unks, total_ws = data2instance2(dev_data, [(word_to_ix,0), (cap_to_ix,-1), (suffix_to_ix,0), (tag_to_ix,-1)])
print "dev size: " + str(len(dev_instances))
print "word unk:",
print unks[0],
print total_ws[0],
print unks[0]*1.0 / total_ws[0]
print "cap unk:",
print unks[1],
print total_ws[1],
print unks[1]*1.0 / total_ws[1]
print "suffix unk:",
print unks[2],
print total_ws[2],
print unks[2]*1.0 / total_ws[2]
print "tag unk:",
print unks[3],
print total_ws[3],
print unks[3]*1.0 / total_ws[3]
###########################################################
# prepare test instance
tst_instances, unks, total_ws = data2instance2(tst_data, [(word_to_ix,0), (cap_to_ix,-1), (suffix_to_ix,0), (tag_to_ix,-1)])
print "tst size: " + str(len(tst_instances))
print "word unk:",
print unks[0],
print total_ws[0],
print unks[0]*1.0 / total_ws[0]
print "cap unk:",
print unks[1],
print total_ws[1],
print unks[1]*1.0 / total_ws[1]
print "suffix unk:",
print unks[2],
print total_ws[2],
print unks[2]*1.0 / total_ws[2]
print "tag unk:",
print unks[3],
print total_ws[3],
print unks[3]*1.0 / total_ws[3]
print "GPU", use_cuda
if use_cuda:
    bilstm = bilstm.cuda()

trainIters(trn_instances, dev_instances, tst_instances, bilstm, print_every=1000, evaluate_every=10000, learning_rate=LEARNING_RATE)

