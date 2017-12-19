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


class AttnDecoderRNN(nn.Module):
    def __init__(self,  mask_pool, tags_info, tag_dim, feat_dim, encoder_hidden_dim, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.tags_info = tags_info
        self.mask_pool = mask_pool
        self.tag_dim = tag_dim
        self.feat_dim = feat_dim
        self.hidden_dim = encoder_hidden_dim * 2

        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)
        self.tag_embeds = nn.Embedding(self.tags_info.tag_size, self.tag_dim)

        self.lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers= self.n_layers)

        self.feat = nn.Linear(self.hidden_dim + self.tag_dim, self.feat_dim)
        self.feat_tanh = nn.Tanh()
        self.out = nn.Linear(self.feat_dim, self.tags_info.tag_size)

    def forward(self, input, hidden, encoder_outputs, mask_variable, train=True):

        if train:
            self.lstm.dropout = self.dropout_p
            embedded = self.tag_embeds(input).unsqueeze(1)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)

            attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_outputs.transpose(0,1).transpose(1,2)).view(output.size(0),-1))
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.transpose(0,1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(output.size(0),-1)))

            global_score = self.out(feat_hiddens)

            output = F.log_softmax(global_score + (mask_variable - 1) * 1e10)
            return output
        else:
            self.lstm.dropout = 0.0
            tokens = []
            self.mask_pool.reset()
            while True:
                mask = self.mask_pool.get_step_mask()
                mask_variable = Variable(torch.FloatTensor(mask), requires_grad = False, volatile=True).unsqueeze(0)
                if use_cuda:
                    mask_variable = mask_variable.cuda()
                embedded = self.tag_embeds(input).view(1, 1, -1)
                output, hidden = self.lstm(embedded, hidden)

                attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1))
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

                global_score = self.out(feat_hiddens)

                output = global_scorglobal_scoree + (mask_variable - 1) * 1e10

                _, input = torch.max(output,1)
                idx = input.view(-1).data.tolist()[0]

                tokens.append(idx)
                self.mask_pool.update( idx)

                if idx == tags_info.tag_to_ix[tags_info.EOS]:
                    break
            return Variable(torch.LongTensor(tokens), volatile=True)
		
    def initC(self, back_prop=True):
        if use_cuda:
            result = Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()
            if back_prop == False:
                result.volatile=True
            return result
        else:
            result = Variable(torch.zeros(1, 1, self.hidden_dim))
            if back_prop == False:
                result.volatile=True
            return result

def train(sentence_variable, target_variable, gold_variable, mask_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, back_prop=True):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    sentence_length = sentence_variable[0].size(0)
   
    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)

    assert sentence_length == len(target_variable) and sentence_length == len(mask_variable)

    idx = 0
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

def trainIters(trn_instances, dev_instances, encoder, decoder, print_every=100, evaluate_every=1000, learning_rate=0.001):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate, weight_decay=1e-4)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate, weight_decay=1e-4)

    criterion = nn.NLLLoss()

    masks = []

    for instance in trn_instances:
        #print "=========="
        token_masks = []
        for acts in instance[-1]:
            #print acts
            decoder.mask_pool.reset()
            token_masks.append(decoder.mask_pool.get_all_mask(acts))
        masks.append(token_masks)

    idx = -1
    iter = 0
    while True:
        idx += 1
        iter += 1
        if idx == len(trn_instances):
            idx = 0

        sentence_variable = []
        target_variable = []
        mask_variable = []
        gold_variable = []

        if use_cuda:
            sentence_variable.append(Variable(trn_instances[idx][0]).cuda())
            sentence_variable.append(Variable(trn_instances[idx][1]).cuda())
            sentence_variable.append(Variable(trn_instances[idx][2]).cuda())
            target_variable = [ Variable(torch.LongTensor(x)).cuda() for x in trn_instances[idx][-1] ]
            gold_variable = [ Variable(torch.LongTensor(x)).cuda() for x in trn_instances[idx][-1] ]
            mask_variable = [ Variable(torch.FloatTensor(mask), requires_grad = False).cuda() for mask in masks[idx] ]
        else:
            sentence_variable.append(Variable(trn_instances[idx][0]))
            sentence_variable.append(Variable(trn_instances[idx][1]))
            sentence_variable.append(Variable(trn_instances[idx][2]))
            target_variable = [ Variable(torch.LongTensor(x)) for x in trn_instances[idx][-1] ]
            gold_variable = [ Variable(torch.LongTensor(x)) for x in trn_instances[idx][-1] ]
            mask_variable = [ Variable(torch.FloatTensor(mask), requires_grad = False) for mask in masks[idx] ]

        loss = train(sentence_variable, target_variable, gold_variable, mask_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch %.6f : %.10f' % (iter*1.0 / len(trn_instances), print_loss_avg))

        if iter % evaluate_every == 0:
            evaluate(dev_instances, encoder, decoder, str(int(iter/evaluate_every)))

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

from utils import readfile
from utils import readpretrain
from utils import data2instance
from tag import Tag

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

trn_data = readfile(trn_file)
word_to_ix = {UNK:0}
pos1_to_ix = {UNK:0}
pos2_to_ix = {UNK:0}
for sentence, _, postags1, postags2, actions in trn_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for postag in postags1:
        if postag not in pos1_to_ix:
            pos1_to_ix[postag] = len(pos1_to_ix)
    for postag in postags2:
        if postag not in pos2_to_ix:
            pos2_to_ix[postag] = len(pos2_to_ix)

##############################################
## Tags
tags_info = Tag(tag_info_file)
mask_pool = SimpleMask(tags_info)

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

WORD_EMBEDDING_DIM = 64
PRETRAIN_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 32

TAG_DIM = 128
INPUT_DIM = 100
ENCODER_HIDDEN_DIM = 256
FEAT_DIM = 128

encoder = EncoderRNN(len(word_to_ix), WORD_EMBEDDING_DIM, len(pretrain_to_ix), PRETRAIN_EMBEDDING_DIM, torch.FloatTensor(pretrain_embeddings), len(pos1_to_ix), POS_EMBEDDING_DIM, INPUT_DIM, ENCODER_HIDDEN_DIM, n_layers=2, dropout_p=0.1)
attn_decoder = AttnDecoderRNN(mask_pool, tags_info, TAG_DIM, FEAT_DIM, ENCODER_HIDDEN_DIM, n_layers=1, dropout_p=0.1)

###########################################################
# prepare training instance
trn_instances = data2instance(trn_data, [(word_to_ix,0), (pretrain_to_ix,0), (pos1_to_ix,0), (pos2_to_ix,0), (tags_info.tag_to_ix,-1)])
print "trn size: " + str(len(trn_instances))
###########################################################
# prepare development instance
dev_instances = data2instance(dev_data, [(word_to_ix,0), (pretrain_to_ix,0), (pos1_to_ix,0), (pos2_to_ix,0), (tags_info.tag_to_ix,-1)])
print "dev size: " + str(len(dev_instances))
###########################################################
# prepare test instance
tst_instances = data2instance(tst_data, [(word_to_ix,0), (pretrain_to_ix,0), (pos1_to_ix,0), (pos2_to_ix,0), (tags_info.tag_to_ix,-1)])
print "tst size: " + str(len(tst_instances))

print "GPU", use_cuda
if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()

trainIters(trn_instances, dev_instances, encoder, attn_decoder, print_every=10, evaluate_every=50000, learning_rate=0.01)

