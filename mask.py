


class SimpleMask:
	## 1) bracket completed, ensuring left bracket "(" and right bracket ")" appear pairly.
	## 2) relation(variables, variables) or relation(variables)
	def __init__(self, tags_info):
		self.tags_info = tags_info
		self.reset()
		self.mask = 0
		self.need = 1
		self.MAX_TAG = 200

	def reset(self):
		self.tag_count = 0
		self.stack = [-2]

	def get_all_mask(self, inputs):
		res = []
		#self._print_state()
		res.append(self.get_step_mask())
		#print res[-1]
		for ix in inputs:
			#print ix
			assert res[-1][ix] != self.mask
			self.update(ix)
			#self._print_state()
			res.append(self.get_step_mask())
			#print res[-1]
		return res

	def get_step_mask(self):
		if self.stack[-1] == -1:
			re = self._get_zeros(self.tags_info.tag_size)
			if len(self.stack) == 2:
				re[self.tags_info.tag_to_ix[self.tags_info.EOS]] = self.need
				if self.tag_count <= self.MAX_TAG:
					re[self.tags_info.tag_to_ix[self.tags_info.COMBINE_F]] = self.need
					re[self.tags_info.tag_to_ix[self.tags_info.COMBINE_B]] = self.need
			else:
				if self.tag_count > self.MAX_TAG:
					re[self.tags_info.tag_to_ix[self.tags_info.REDUCE]] = self.need
				else:
					re[self.tags_info.tag_to_ix[self.tags_info.COMBINE_F]] = self.need
					re[self.tags_info.tag_to_ix[self.tags_info.COMBINE_B]] = self.need
					re[self.tags_info.tag_to_ix[self.tags_info.REDUCE]] = self.need
			return re
		elif self.stack[-1] == -2:
			re = self._get_ones(self.tags_info.tag_size)
			re[self.tags_info.tag_to_ix[self.tags_info.SOS]] = self.mask
			re[self.tags_info.tag_to_ix[self.tags_info.EOS]] = self.mask
			re[self.tags_info.tag_to_ix[self.tags_info.COMBINE_B]] = self.mask
			re[self.tags_info.tag_to_ix[self.tags_info.COMBINE_F]] = self.mask
			re[self.tags_info.tag_to_ix[self.tags_info.REDUCE]] = self.mask
				
			return re
		else:
			assert False
		
	def update(self, ix):
		assert ix < self.tags_info.tag_size
		if ix == self.tags_info.tag_to_ix[self.tags_info.REDUCE]:
			self.stack.pop()
			self.stack.pop()
		elif ix == self.tags_info.tag_to_ix[self.tags_info.COMBINE_F] or ix == self.tags_info.tag_to_ix[self.tags_info.COMBINE_B]:
			self.stack.append(-2)
			self.tag_count += 1
		else:
			self.stack.append(-1)
			self.tag_count += 1
				
	def _print_state(self):
		print "tag_count", self.tag_count
		print "stack", self.stack
	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]




