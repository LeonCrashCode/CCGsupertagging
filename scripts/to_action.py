import sys

def produce(tag):
	actions = []
	prev = ""
	for i in range(len(tag)):
		if tag[i] == "(":
			if prev != "":
				actions.append(prev)
				prev = ""
		elif tag[i] in ["/", "\\"]:
			if prev != "":
				actions.append(prev)
				prev = ""
			actions.append(tag[i])
		elif tag[i] == ")":
			if prev != "":
				actions.append(prev)
				prev = ""
			actions.append("reduce")
		else:
			prev += tag[i]

	return actions

for line in open(sys.argv[1]):
	line = line.strip()
	if line == "":
		print
		continue
	line = line.split()
	newline = line[0:-1]
	if ("/" not in line[-1]) and "\\" not in line[-1]:
		tag = "(TOP/"+line[-1]+")"
	else:
		tag = "(TOP/("+line[-1]+"))"
	actions = produce(tag)
	newline.append(" ".join(actions))
	print "\t".join(newline)

