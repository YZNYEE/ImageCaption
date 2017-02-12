require'nn'
require'nngraph'
require'torch'
local utils = require'misc.utils'


local layer, parent = torch.class('nn.SenInfo','nn.Module')
function layer:__init(opt)

	parent.__init(self)
	-- print(111111111111)

	self.vocab_size = utils.getopt(opt, 'vocab_size', nil)
	self.encoding_size = utils.getopt(opt, 'encoding_size', 512)
	self.seq_length = utils.getopt(opt, 'seq_length', 16)
	self.batch_size = utils.getopt(opt, 'batch_size', nil)
	assert(self.vocab_size ~= nil,'vocab_size error')
	self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.encoding_size)
	-- self:initialWeight()
	-- print(self.lookup_table)

end

function layer:createClones(length)

	print('create clones inside the SenInfo')
	self.lookup_tables = {[0] = self.lookup_table}
	for i=2,self.length+1 do
		self.lookup_tables[i] = self.lookup_table:clone('weight', 'gradweight')
	end

end

function layer:initialWeight()

	self.weight = self.lookup_table.weight
	self.gradWeight = self.lookup_table.gradWeight

end

-- inputs is DxN LongTensor. D is batch_size. N is the length
function layer:updateOutput(inputs)

	self.size = inputs:size()
	self.output = nil

	self.inputs = self.lookup_table:forward(inputs)
	self.output = torch.FloatTensor(self.batch_size, self.encoding_size):zero():type(self._type)

	for i=1,self.batch_size do
		self.output[i] = self.inputs[i]:mean(1)
	end

	return self.output

end



function layer:updateGradInput(inputs, gradOutput)

	local gout = gradOutput:div(self.size[2])
	local dlookup_table = torch.FloatTensor(self.batch_size, self.seq_length, self.encoding_size):zero():type(self._type)
	for j = 1,self.batch_size do dlookup_table[j] = torch.expand(gout[j]:resize(1, self.encoding_size), self.seq_length, self.encoding_size) end
	self.lookup_table:backward(inputs, dlookup_table)

	return torch.Tensor()

end

function layer:getModuleList()

	return {self.lookup_table}

end

function layer:changeBatchsize(batch_size)
	self.batch_size = batch_size
end

function layer:parameters()

	local p1,g1 = self.lookup_table:parameters()
	local params={}
	local grad_params={}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(g1) do table.insert(grad_params, v) end

	return params, grad_params

end

function layer:clone(...)

    local f = torch.MemoryFile("rw"):binary()
    f:writeObject(self)
    f:seek(1)
    local clone = f:readObject()
    f:close()

    clone.lookup_table = self.lookup_table:clone(...)

    return clone

end

