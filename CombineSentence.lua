require'nn'
require'nngraph'
require'torch'

local layer, parent = torch.class('nn.SenInfo','nn.module')
local utils = require'misc.utils'

function layer:_init(opt)

	self.vocab_size = utils.getopt(opt, 'vocab_size', nil)
	self.encoding_size = utils.getopt(opt, 'encoding_size', 512)
	self.length = utils.getopt(opt, 'length', 16)
	assert(self.vocab_size ~= nil,'vocab_size error')
	self.lookup_table = nn.LookupTable(self.vocab_size+1, self.encoding_size)
	createClones(self.length)

end

function layer:createClones(length)

	print('create clones inside the SenInfo')
	self.lookup_tables = {[0] = self.lookup_table}
	for i=2,self.length+1 do
		self.lookup_tables[i] = self.lookup_table:clone('weight', 'gradweight')
	end

end

function layer:updataOutput(inputs)

	self.size = inputs:size()
	self.inputs = {}
	self.output = nil

	for i=1,size[1] do
		self.inputs[i]=self.lookup_tables[i]:forward(inputs[i])
	end

	self.output = torch.FloatTensor(self.encoding_size):zero()
	for i=1,size[1] do
		self.output:add(self.inputs[i])
	end

	self.output:div(self.size[1])
	return self.output

end

function layer:updataGradInput(inputs, gradOutput)

	gradOutput:div(self.size[1])
	for i=1,self.size[1] do
		self.lookup_tables[i]:backward(inputs[i],gradOutput)
	end

end

