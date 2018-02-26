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

def data2instance3(trn_data, ixes):
	instances = []
	for one in trn_data:
		instances.append([])
		instances[-1].append([get_from_ix(w, ixes[0], 0) for w in one[0]])
		instances[-1].append([])
		for w in one[0]:
			unk = unkized(w, ixes[0])
			assert unk in ixes[0]
			instances[-1][-1].append(ixes[0][unk])
		instances[-1].append([get_from_ix(w, ixes[1], 0) for w in one[1]])
		instances[-1].append([get_from_ix(w, ixes[2], 0) for w in one[2]])
		instances[-1].append([get_from_ix(w, ixes[3], 0) for w in one[3]])
		instances[-1].append([get_from_ix(w, ixes[4], 0) for w in one[4]])
	return instances

def cap(w):
	if isupper(w[0]):
		return "CAP"
	else:
		return "UCAP"
def data2instance4(trn_data, ixes):
        instances = []
        for one in trn_data:
                instances.append([])
                instances[-1].append([get_from_ix(w, ixes[0], 0) for w in one[0]])
                instances[-1].append([])
                for w in one[0]:
                        unk = unkized(w, ixes[0])
                        assert unk in ixes[0]
                        instances[-1][-1].append(ixes[0][unk])
                instances[-1].append([get_from_ix(w, ixes[1], 0) for w in one[1]])
                instances[-1].append([get_from_ix(w, ixes[2], 0) for w in one[2]])
                instances[-1].append([get_from_ix(w, ixes[3], 0) for w in one[3]])
		instances[-1].append([get_from_ix("%5s" % w[0:5], ixes[4], 0) for w in one[0]])
		instances[-1].append([get_from_ix("%-5s" % w[-5:0], ixes[4], 0) for w in one[0]])
		instances[-1].append([get_from_ix(cap(w), ixes[5], 0) for w in one[0]])
                instances[-1].append([get_from_ix(w, ixes[6], 0) for w in one[4]])
        return instances

def islower(c):
	if c >= 'a' and c <= 'z':
		return True
	return False
def isupper(c):
	if c >= 'A' and c <= 'Z':
		return True
	return False

def isdigital(c):
	if c >= '0' and c <= '9':
		return True
	return False

def isalpha(c):
	if islower(c) or isupper(c):
		 return True
	return False

def unkized(w, word_to_ix):
	numCaps = 0
	hasDigit = False
	hasDash = False
	hasLower = False

	result = "UNK"
	for c in w:
		if isdigital(c):
			hasDigit = True
		elif c == '-':
			hasDash = True
		elif islower(c):
			hasLower = True
		elif isupper(c):
			numCaps += 1
	lower = w.lower()
	if isupper(w[0]):
		if numCaps == 1:
			result = result + "-INITC"
			if lower in word_to_ix:
				result = result + "-KNOWNLC"
		else:
			result = result + "-CAPS"
	elif isalpha(w[0]) == False and numCaps > 0:
		result = result + "-CAPS"
	elif hasLower:
		result = result + "-LC"

	if hasDigit:
		result = result + "-NUM"
	if hasDash:
		result = result + "-DASH"
	if lower[-1] == 's' and len(lower) >= 3:
		ch2 = '0'
		if len(lower) >= 2:
			ch2 = lower[-2]
		if ch2 != 's' and ch2 != 'i' and ch2 != 'u':
			result = result + "-s"
	elif len(lower) >= 5 and hasDash == False and (hasDigit == False or numCaps == 0):
		ch1 = '0'
		ch2 = '0'
		ch3 = '0'
		if len(lower) >= 1:
			ch1 = lower[-1]
		if len(lower) >= 2:
			ch2 = lower[-2]
		if len(lower) >= 3:
			ch3 = lower[-3]
		if ch2 == 'e' and ch1 == 'd':
			result = result + "-ed"
		elif ch3 == 'i' and ch2 == 'n' and ch1 == 'g':
			result = result + "-ing"
		elif ch3 == 'i' and ch2 == 'o' and ch1 == 'n':
			result = result + "-ion"
		elif ch2 == 'e' and ch1 == 'r':
			result = result + "-er"
		elif ch3 == 'e' and ch2 == 's' and ch1 == 't':
			result = result + "-est"
		elif ch2 == 'l' and ch1 == 'y':
			result = result + "-ly"
		elif ch3 == 'i' and ch2 == 't' and ch1 == 'y':
			result = result + "-ity"
		elif ch1 == 'y':
			result = result + "-y"
		elif ch2 == 'a' and ch1 == 'l':
			result = result + "-al"
	return result

def all_possible_UNK():
	results_one = []
	results_two = []
	results_three = []
	results_four = []

	results_one.append("UNK")
	results_one.append("UNK-INITC")
	results_one.append("UNK-INITC-KNOWNLC")
	results_one.append("UNK-CAPS")
	results_one.append("UNK-LC")
	
	for item in results_one:
		results_two.append(item+"-NUM")
	for item in results_one:
		results_three.append(item+"-DASH")
	for item in results_two:
		results_four.append(item+"-DASH")

	results_five = results_one + results_two + results_three + results_four

	results = []
	for item in results_five:
		results.append(item+"-s")
		results.append(item+"-ed")
		results.append(item+"-ing")
		results.append(item+"-ion")
		results.append(item+"-er")
		results.append(item+"-est")
		results.append(item+"-ly")
		results.append(item+"-ity")
		results.append(item+"-y")
		results.append(item+"-al")
	return results + results_five
	
