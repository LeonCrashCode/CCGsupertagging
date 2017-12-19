import sys
import re

p = re.compile("(<L .*?>)")

for line in open(sys.argv[1]):
	line = line.strip()
	if line[0:2] == "ID":
		continue
	matchs = p.findall(line)
	for item in matchs:
		item = item[2:-1].split()
		print "\t".join([item[-2], item[1], item[2], item[0]])
	print 