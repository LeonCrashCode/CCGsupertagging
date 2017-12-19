import sys

L = {}
for filename in sys.argv[1:]:
	for line in open(filename):
		line = line.strip()
		if line == "":
			continue
		line = line.split()[-1]
		if line in L:
			L[line] += 1
		else:
			L[line] = 1

for key in L.keys():
	if L[key] >= 1:
		print key