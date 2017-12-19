
class Tag:
	def __init__(self, filename):
		self.filename = filename

		self.SOS = "<SOS>"
		self.EOS = "<EOS>"
		self.COMBINE_F = "/"
		self.COMBINE_B = "\\"
		self.REDUCE = "reduce"
		
		self.tag_to_ix = {self.SOS:0, self.EOS:1, self.COMBINE_F:2, self.COMBINE_B:3, self.REDUCE:4}
		self.ix_to_tag = [self.SOS, self.EOS, self.COMBINE_F, self.COMBINE_B, self.REDUCE]

		for line in open(filename):
			line = line.strip()
			if line[0] == "#":
				continue
			if line not in self.tag_to_ix:
				self.tag_to_ix[line] = len(self.tag_to_ix)
				self.ix_to_tag.append(line)

		self.tag_size = len(self.tag_to_ix)

		



