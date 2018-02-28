# -*- coding: utf-8 -*-
##
#   Hierachical Decoder
#   200 glove pretrained
#   adding windows
#   re-init parameter
#   no pos
#   capt
#   char
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
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    device = int(sys.argv[1])
    torch.cuda.manual_seed_all(12345678)

torch.manual_seed(12345678)

dev_out_dir = sys.argv[2]+"_dev/"
tst_out_dir = sys.argv[2]+"_tst/"
model_dir = sys.argv[2]+"_model/"
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"
word_to_ix = {UNK:0}
ix_to_word = [UNK]
pos1_to_ix = {UNK:0}
ix_to_pos1 = [UNK]
pos2_to_ix = {UNK:0}
ix_to_pos2 = [UNK]
tag_to_ix = {UNK:0, SOS:1, EOS:2}
ix_to_tag = [UNK, SOS, EOS]
char_to_ix = {UNK:0}
ix_to_char = [UNK]
cap_to_ix = {"CAP":1, "UCAP":0}
ix_to_cap = ["UCAP", "CAP"]

word_to_cnt = {}
rare_word_ix = []
tag_size = 0

WORD_EMBEDDING_DIM = 128
PRETRAIN_EMBEDDING_DIM = 200
POS_EMBEDDING_DIM = 128
CHAR_EMBEDDING_DIM = 32
CAP_EMBEDDING_DIM = 32
INPUT_DIM = 256
ENCODER_HIDDEN_DIM = 512
FEAT_DIM = 256

class EncoderRNN(nn.Module):
    def __init__(self, word_size, word_dim, pretrain_size, pretrain_dim, pretrain_embeddings, pos_size, pos_dim, char_size, char_dim, cap_size, cap_dim, input_dim, hidden_dim, feat_dim, n_layers=1, dropout_p=0.0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(word_size, word_dim)
        self.pretrain_embeds = nn.Embedding(pretrain_size, pretrain_dim)
        self.pretrain_embeds.weight = nn.Parameter(pretrain_embeddings, False)
        self.pos_embeds = nn.Embedding(pos_size, pos_dim)
        self.char_embeds = nn.Embedding(char_size, char_dim)
	self.cap_embeds = nn.Embedding(cap_size, cap_dim)
	self.dropout = nn.Dropout(self.dropout_p)

        self.embeds2input = self.linear_init(nn.Linear(word_dim + pretrain_dim + char_dim*2+ cap_dim, input_dim))
	self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_dim*3, hidden_dim, num_layers=self.n_layers, bidirectional=True)
        for name, param in self.lstm.named_parameters():
	    if "weight" in name:
	    	tmp = torch.nn.init.orthogonal(param)

    def forward(self, sentence, hidden, train=True):
        word_embedded = self.word_embeds(sentence[0])
        pretrain_embedded = self.pretrain_embeds(sentence[1])
        pos_embedded = self.pos_embeds(sentence[2])
	left_char_embedded = self.char_embeds(sentence[3])
	right_char_embedded = self.char_embeds(sentence[4])
	cap_embedded = self.cap_embeds(sentence[5])

        if train:
            word_embedded = self.dropout(word_embedded)
            pos_embedded = self.dropout(pos_embedded)
	    left_char_embedded = self.dropout(left_char_embedded)
	    right_char_embedded = self.dropout(right_char_embedded)
	    cap_embedded = self.dropout(cap_embedded)
            self.lstm.dropout = self.dropout_p

        embeds = self.tanh(self.embeds2input(torch.cat((word_embedded, pretrain_embedded, left_char_embedded, right_char_embedded, cap_embedded), 1))).view(len(sentence[0]),1,-1)
        ##windows
	begin_padding = self.initPadding()
	end_padding = self.initPadding()
	windows = []
	if len(sentence[0]) == 1:
	    windows.append(torch.cat((begin_padding[0], embeds[0], end_padding[0]), 1))
	else:
	    for i in range(len(sentence[0])):
	    	if i == 0:
		    windows.append(torch.cat((begin_padding[0], embeds[i], embeds[i+1]), 1))
		elif i == len(sentence[0])-1:
		    windows.append(torch.cat((embeds[i-1], embeds[i], end_padding[0]), 1))
		else:
		    windows.append(torch.cat((embeds[i-1], embeds[i], embeds[i+1]), 1))
	inputs = torch.cat(windows, 0).unsqueeze(1)
		
	output, hidden = self.lstm(inputs, hidden)
	return output, hidden

    def linear_init(self, linear):
	fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(linear.weight)
	linear.weight = torch.nn.Parameter((torch.nn.init.normal(linear.weight, 0, 1.0 / math.sqrt(fan_in))*0.1).data)
        linear.bias = torch.nn.Parameter((torch.nn.init.normal(linear.bias, 0, 1)*0.1).data)
	return linear
	
    def initHidden(self):
        if use_cuda:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(device),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(device))
            return result
        else:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)))
	    return result
    def initPadding(self):
	if use_cuda:
	    result = Variable(torch.zeros(1, 1, self.input_dim)).cuda(device)
	    return result
	else:
	    result = Variable(torch.zeros(1, 1, self.input_dim))
	    return result
class DecoderRNN(nn.Module):
    def __init__(self, input_dim, hidden,  n_layers=1, dropout_p=0.0):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, sentence, hidden, train=True):

        if train:
            word_embedded = self.dropout(word_embedded)
            pos_embedded = self.dropout(pos_embedded)
	    left_char_embedded = self.dropout(left_char_embedded)
	    right_char_embedded = self.dropout(right_char_embedded)
	    cap_embedded = self.dropout(cap_embedded)
            self.lstm.dropout = self.dropout_p

        embeds = self.tanh(self.embeds2input(torch.cat((word_embedded, pretrain_embedded, left_char_embedded, right_char_embedded, cap_embedded), 1))).view(len(sentence[0]),1,-1)
        ##windows
	begin_padding = self.initPadding()
	end_padding = self.initPadding()
	windows = []
	if len(sentence[0]) == 1:
	    windows.append(torch.cat((begin_padding[0], embeds[0], end_padding[0]), 1))
	else:
	    for i in range(len(sentence[0])):
	    	if i == 0:
		    windows.append(torch.cat((begin_padding[0], embeds[i], embeds[i+1]), 1))
		elif i == len(sentence[0])-1:
		    windows.append(torch.cat((embeds[i-1], embeds[i], end_padding[0]), 1))
		else:
		    windows.append(torch.cat((embeds[i-1], embeds[i], embeds[i+1]), 1))
	inputs = torch.cat(windows, 0).unsqueeze(1)
		
	output, hidden = self.lstm(inputs, hidden)
	return output, hidden

    def linear_init(self, linear):
	fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(linear.weight)
	linear.weight = torch.nn.Parameter((torch.nn.init.normal(linear.weight, 0, 1.0 / math.sqrt(fan_in))*0.1).data)
        linear.bias = torch.nn.Parameter((torch.nn.init.normal(linear.bias, 0, 1)*0.1).data)
	return linear
	
    def initHidden(self):
        if use_cuda:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(device),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(device))
            return result
        else:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)))
	    return result
    def initPadding(self):
	if use_cuda:
	    result = Variable(torch.zeros(1, 1, self.input_dim)).cuda(device)
	    return result
	else:
	    result = Variable(torch.zeros(1, 1, self.input_dim))
	    return result
def train(sentence_variable, gold_variable, bilstm, decoder, bilstm_optimizer, decoder_optimizer, criterion, back_prop=True):
    bilstm_hidden = bilstm.initHidden()
    sentence_length = sentence_variable[0].size(0)
    bilstm_output, bilstm_hidden = bilstm(sentence_variable, bilstm_hidden)
    
    for i in range(bilstm_output.size(0)):
	decoder_output = decoder(bilstm_output[i], bilstm_hidden[i], gold_variable[i])
    	EOS_var = Variable(torch.LongTensor([2]))
    	if use_cuda:
	    EOS_var = EOS_var.cuda(device)
	target_variable = torch.cat((gold_variable[i], EOS_var))
    	loss += criterion(decoder_output, target_variable) / target_variable.size(0)

    if back_prop == True:
	bilstm_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
        loss.backward()
        bilstm_optimizer.step()
	decoder_optimizer.step()
    
    return loss / sentence_length 

def decode(sentence_variable, bilstm):
    bilstm_hidden = bilstm.initHidden()
    bilstm_output = bilstm(sentence_variable, bilstm_hidden)
    scores, indexs = torch.max(bilstm.out(bilstm_output), 1)
    return indexs.view(-1).data.tolist()

def trainIters(trn_instances, dev_instances, tst_instances, bilstm, print_every=100, evaluate_every=1000, learning_rate=0.001):
    print_loss_total = 0  # Reset every print_every

    #bilstm_optimizer = optim.Adam(filter(lambda p: p.requires_grad, bilstm.parameters()), lr=learning_rate, weight_decay=1e-4)
    bilstm_optimizer = optim.SGD(filter(lambda p: p.requires_grad, bilstm.parameters()), lr=learning_rate)

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
            sentence_variable.append(Variable(torch.LongTensor(input_words)).cuda(device)) # words
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][2])).cuda(device)) #pretrain
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][3])).cuda(device)) #pos
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][5])).cuda(device)) #leftchar
	    sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][6])).cuda(device)) #rightchar
	    sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][7])).cuda(device)) #cap
	    for tag in trn_instances[idx][-1]:
		gold_variable.append(Variable(torch.LongTensor(tag)).cuda(device)) 
        else:
            sentence_variable.append(Variable(torch.LongTensor(input_words)))
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][2])))
            sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][3])))
	    sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][5])))
	    sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][6])))
	    sentence_variable.append(Variable(torch.LongTensor(trn_instances[idx][7])))
	    for tag in trn_instances[idx][-1]:
                gold_variable.append(Variable(torch.LongTensor(tag)))            

        loss = train(sentence_variable, gold_variable, bilstm, bilstm_optimizer, criterion, True)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch %.6f : %.10f' % (iter*1.0 / len(trn_instances), print_loss_avg))

        if iter % evaluate_every == 0:
	    """
	    dev_loss = 0
	    for instance in dev_instances:
		#for i in range(len(instance[0])):
		#	print ix_to_word[instance[0][i]], ix_to_pretrain[instance[1][i]], ix_to_pos1[instance[2][i]]
		#exit(1)
		dev_sentence_variable = []
		dev_gold_variable = []
		if use_cuda:
		    dev_sentence_variable.append(Variable(instance[0], volatile=True).cuda(device))
		    dev_sentence_variable.append(Variable(instance[1], volatile=True).cuda(device))
		    dev_sentence_variable.append(Variable(instance[2], volatile=True).cuda(device))
		    dev_gold_variable = Variable(instance[-1], volatile=True).cuda(device)
		else:
		    dev_sentence_variable.append(Variable(instance[0], volatile=True))
		    dev_sentence_variable.append(Variable(instance[1], volatile=True))
		    dev_sentence_variable.append(Variable(instance[2], volatile=True))
		    dev_gold_variable = Variable(instance[-1], volatile=True)
		dev_loss += train(dev_sentence_variable, dev_gold_variable, bilstm, None, criterion, False)
            """
	    dev_acc = evaluate(dev_instances, bilstm, dev_out_dir+str(int(iter/evaluate_every))+".pred")
            tst_acc = evaluate(tst_instances, bilstm, tst_out_dir+str(int(iter/evaluate_every))+".pred")
	    #print('dev %d loss %.10f, dev_acc: %.10f, tst_acc: %.10f' %(int(iter/evaluate_every), dev_loss/len(dev_instances), dev_acc, tst_acc))
	    print('dev %d, dev_acc: %.10f, tst_acc: %.10f' %(int(iter/evaluate_every), dev_acc, tst_acc))
def evaluate(instances, bilstm, dir_path):
    out = open(dir_path,"w")
    correct = 0
    total = 0
    for instance in instances:
        input_words = []
        for i in range(len(instance[0])):
            w = instance[0][i]
            if w == 0:
                input_words.append(instance[1][i])
            else:
                input_words.append(instance[0][i])

        sentence_variable = []
        if use_cuda:
            sentence_variable.append(Variable(torch.LongTensor(input_words), volatile=True).cuda(device))
            sentence_variable.append(Variable(torch.LongTensor(instance[2]), volatile=True).cuda(device))
            sentence_variable.append(Variable(torch.LongTensor(instance[3]), volatile=True).cuda(device))
            sentence_variable.append(Variable(torch.LongTensor(instance[5]), volatile=True).cuda(device))
	    sentence_variable.append(Variable(torch.LongTensor(instance[6]), volatile=True).cuda(device))
	    sentence_variable.append(Variable(torch.LongTensor(instance[7]), volatile=True).cuda(device))
	else:
            sentence_variable.append(Variable(torch.LongTensor(input_words), volatile=True))
            sentence_variable.append(Variable(torch.LongTensor(instance[2]), volatile=True))
            sentence_variable.append(Variable(torch.LongTensor(instance[3]), volatile=True))
	    sentence_variable.append(Variable(torch.LongTensor(instance[5]), volatile=True))
	    sentence_variable.append(Variable(torch.LongTensor(instance[6]), volatile=True))
	    sentence_variable.append(Variable(torch.LongTensor(instance[7]), volatile=True))
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

from utils import readfile2
from utils import readpretrain
from utils import data2instance4
from utils import all_possible_UNK

trn_file = "train.input"
dev_file = "dev.input"
tst_file = "test.input"
pretrain_file = "sskip.100.vectors"
pretrain_file = "glove.6B.200d.txt"
#trn_file = "train.input.part"
#dev_file = "train.input.part"
#tst_file = "test.actions.part"
#pretrain_file = "sskip.100.vectors.part"

trn_data = readfile2(trn_file)
for sentence, _, postags1, postags2, tags in trn_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
	    ix_to_word.append(word)
        if word not in word_to_cnt:
            word_to_cnt[word] = 1
        else:
            word_to_cnt[word] += 1
	left = "%5s" % word[0:5]
	right = "%-5s" % word[-5:]
	if left not in char_to_ix:
	    char_to_ix[left] = len(char_to_ix)
	    ix_to_char.append(left)
	if right not in char_to_ix:
	    char_to_ix[right] = len(char_to_ix)
	    ix_to_char.append(right)

    for postag in postags1:
        if postag not in pos1_to_ix:
            pos1_to_ix[postag] = len(pos1_to_ix)
	    ix_to_pos1.append(postag)
    for postag in postags2:
        if postag not in pos2_to_ix:
            pos2_to_ix[postag] = len(pos2_to_ix)
	    ix_to_pos2.append(postag)
    for tag in tags:
	for tt in tag:
            if tt not in tag_to_ix:
            	tag_to_ix[tt] = len(tag_to_ix)
            	ix_to_tag.append(tt)
tag_size = len(tag_to_ix)

###rare_word
for key in word_to_cnt.keys():
    if word_to_cnt[key] == 1:
        rare_word_ix.append(word_to_ix[key])
###
pretrain_to_ix = {UNK:0}
ix_to_pretrain = [UNK]
pretrain_embeddings = [ [0. for i in range(PRETRAIN_EMBEDDING_DIM)] ] # for UNK 
pretrain_data = readpretrain(pretrain_file)
for one in pretrain_data:
    pretrain_to_ix[one[0]] = len(pretrain_to_ix)
    ix_to_pretrain.append(one[0])
    pretrain_embeddings.append([float(a) for a in one[1:]])
print "pretrain dict size:", len(pretrain_to_ix)

dev_data = readfile2(dev_file)
for _, _, _, _, tags in dev_data:
    for tag in tags:
	for tt in tag:
            if tt not in tag_to_ix:
            	tag_to_ix[tt] = len(tag_to_ix)
            	ix_to_tag.append(tt)
tst_data = readfile2(tst_file)
for _, _, _, _, tags in tst_data:
    for tag in tags:
	for tt in tag:
            if tt not in tag_to_ix:
            	tag_to_ix[tt] = len(tag_to_ix)
            	ix_to_tag.append(tt)

print "word dict size: ", len(word_to_ix)
print "pos1 dict size: ", len(pos1_to_ix)
print "pos2 dict size: ", len(pos2_to_ix)
print "tag dict size: ", tag_size

for item in all_possible_UNK():
    assert item not in word_to_ix

    word_to_ix[item] = len(word_to_ix)
    ix_to_word.append(item)

bilstm = EncoderRNN(len(word_to_ix), WORD_EMBEDDING_DIM, len(pretrain_to_ix), PRETRAIN_EMBEDDING_DIM, torch.FloatTensor(pretrain_embeddings), len(pos1_to_ix), POS_EMBEDDING_DIM, len(char_to_ix), CHAR_EMBEDDING_DIM, len(cap_to_ix), CAP_EMBEDDING_DIM, INPUT_DIM, ENCODER_HIDDEN_DIM, FEAT_DIM, n_layers=2, dropout_p=0.4)

###########################################################
# prepare training instance
trn_instances = data2instance4(trn_data, [word_to_ix, pretrain_to_ix, pos1_to_ix, pos2_to_ix, char_to_ix, cap_to_ix, tag_to_ix])
print "trn size: " + str(len(trn_instances))
###########################################################
# prepare development instance
dev_instances = data2instance4(dev_data, [word_to_ix, pretrain_to_ix, pos1_to_ix, pos2_to_ix, char_to_ix, cap_to_ix, tag_to_ix])
print "dev size: " + str(len(dev_instances))
###########################################################
# prepare test instance
tst_instances = data2instance4(tst_data, [word_to_ix, pretrain_to_ix, pos1_to_ix, pos2_to_ix, char_to_ix, cap_to_ix, tag_to_ix])
print "tst size: " + str(len(tst_instances))

print "GPU", use_cuda
if use_cuda:
    bilstm = bilstm.cuda(device)

trainIters(trn_instances, dev_instances, tst_instances, bilstm, print_every=1000, evaluate_every=10000, learning_rate=0.02)

