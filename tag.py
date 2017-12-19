
class Tag:
	def __init__(self, filename):
		self.filename = filename

		self.SOS = "<SOS>"
		self.EOS = "<EOS>"
		self.TOP = "<TOP>"
		self.combine_f = "/"
		self.combine_b = "\\"
		self.reduce = "reduce"
		
		self.tag_to_ix = {self.SOS:0, self.EOS:1, self.TOP:2, self.combine_f:3, self.combine_b:4, self.reduce:5}
		self.ix_to_tag = [self.SOS, self.EOS, self.TOP, self.combine_f, self.combine_b, self.reduce]

		self.relation_global = list()
		for line in open(filename):
			line = line.strip()
			if line[0] == "#":
				continue
			self.tag_to_ix[line] = len(self.tag_to_ix)
			self.ix_to_tag.append(line)
			
		self.tag_size = len(self.tag_to_ix)

		



