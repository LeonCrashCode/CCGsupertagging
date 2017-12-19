import sys

L = {}
for filename in sys.argv[1:]:
	for line in open(filename):
		line = line.strip()
		if line == "":
			continue
		line = line.split("\t")[-1]
		line = line.split()
		for item in line:
			if item in L:
				L[item] += 1
			else:
				L[item] = 1

for key in L.keys():
	if L[key] >= 1:
		print key