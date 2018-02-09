# -*- coding: utf-8 -*-
import torch
def readfile(filename):
	data = []
	words = []
	pretrains = []
	postags1 = []
	postags2 = []
	actions = []
	with open(filename, "r") as r:
		while True:
			line = r.readline().strip()
			if line == "":
				if len(words) == 0:
					break
				data.append([words, pretrains, postags1, postags2, actions])
				words = []
				pretrains = []
				postags1 = []
				postags2 = []
				actions = []
				continue
			tokens = line.split("\t")
			words.append(tokens[0])
			pretrains.append(tokens[0].lower())
			postags1.append(tokens[1])
			postags2.append(tokens[2])
			actions.append(tokens[3].split())
	return data

def readfile2(filename):
	data = []
	words = []
	pretrains = []
	postags1 = []
	postags2 = []
	tags = []
	with open(filename, "r") as r:
		while True:
			line = r.readline().strip()
			if line == "":
				if len(words) == 0:
					break
				data.append([words, pretrains, postags1, postags2, tags])
				words = []
				pretrains = []
				postags1 = []
				postags2 = []
				tags = []
				continue
			tokens = line.split("\t")
			words.append(tokens[0])
			pretrains.append(tokens[0].lower())
			postags1.append(tokens[1])
			postags2.append(tokens[2])
			tags.append(tokens[3])
	return data

def readpretrain(filename):
	data = []
	with open(filename, "r") as r:
		while True:
			l = r.readline().strip()
			if l == "":
				break
			data.append(l.split())
	return data

def get_from_ix(w, to_ix, unk):
	if w in to_ix:
		return to_ix[w]

	assert unk != -1, "no unk supported"
	return unk

def get_from_ix_list(l, to_ix, unk):
	re = []
	for w in l:
		if w in to_ix:
			re.append(to_ix[w])
		else:
			re.append(unk)
			assert False
	return re

def data2instance(trn_data, ixes):
	instances = []
	for one in trn_data:
		instances.append([])
		for i in range(len(ixes)):
			if i == len(ixes) - 1:
				instances[-1].append([torch.LongTensor(get_from_ix_list(w, ixes[i][0], ixes[i][1])) for w in one[i]])
			else: 
				instances[-1].append(torch.LongTensor([get_from_ix(w, ixes[i][0], ixes[i][1]) for w in one[i]]))
	return instances

def data2instance2(trn_data, ixes):
	instances = []
	for one in trn_data:
		instances.append([])
		for i in range(len(ixes)):
			instances[-1].append(torch.LongTensor([get_from_ix(w, ixes[i][0], ixes[i][1]) for w in one[i]]))
	return instances

