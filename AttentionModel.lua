require'nn'
local utils = require'misc.utils'
local net_utils = require'misc.net_utils'
local GRU = require'misc.GRU'

require 'ConstructAttention'
require 'CombineSentence'

local layer, parent = torch.class('nn.AttentionModel', 'nn.Module')
function layer:__init(opt)

	self.vocab_size = utils.getopt(opt, 'vocab_size')
	self.encoding_size = utils.getopt(opt, 'encoding_size')
	self.get_top_num = utils.getopt(opt, 'get_top_num')
	self.num_of_local_img = utils.getopt(opt, 'num_of_local_img', 1)
	self.rnn_size = utils.getopt(opt, 'rnn_size', 512)
	self.seq_length = utils.getopt(opt, 'seq_length')
	local dropout = utils.getopt(opt, 'dropout', 0)

	assert(self.num_of_local_img == 1, 'its hard')

	self.cstAtt_al = nn.ConstructAttention(opt, 'overall')
    self.cstAtt_lc = nn.ConstructAttention(opt, 'local')

	--print(GRU)
	self.combineSen = nn.SenInfo(opt)
	self.core = GRU.gru(self.encoding_size, self.vocab_size+1, self.rnn_size, 1, dropout)
	self.predict = GRU.predict(self.rnn_size, self.vocab_size+1, dropout)

	self.product = nn.Linear(self.encoding_size, self.encoding_size)
	self.tanh = nn.Tanh()

	self:_createInitState(1)

end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the GRU
  if not self.init_state then self.init_state = {} end -- lazy init
  if self.init_state[1] then
	if self.init_state[1]:size(1) ~= batch_size then
		self.init_state[1]:resize(batch_size, self.rnn_size):zero()
	end
  else
    self.init_state[1] = torch.zeros(batch_size, self.rnn_size)
  end
  self.num_state = #self.init_state

end

function layer:parameters()

	local p1,g1 = self.core:parameters()
	local p2,g2 = self.combineSen.lookup_table:parameters()
	local p3,g3 = self.product:parameters()
	local p4,g4 = self.predict:parameters()

	local params = {}
	for k,v in pairs(p1) do table.insert(params, v) end
	for k,v in pairs(p2) do table.insert(params, v) end
	for k,v in pairs(p3) do table.insert(params, v) end
	for k,v in pairs(p4) do table.insert(params, v) end

	local grad_params = {}
	for k,v in pairs(g1) do table.insert(grad_params, v) end
	for k,v in pairs(g2) do table.insert(grad_params, v) end
	for k,v in pairs(g3) do table.insert(grad_params, v) end
	for k,v in pairs(g4) do table.insert(grad_params, v) end

	return params, grad_params

end

function layer:getModulesList()

	return {self.core, self.combineSen.lookup_table, self.predict, self.product}

end

function layer:evaluate()

	if self.clones == nil then self:createClone() end
	assert( #self.clones == 2)

	self.combineSen:evaluate()
	self.predict:evaluate()
	self.product:evaluate()
	self.tanh:evaluate()
	self.clones[1]:evaluate()
	self.clones[2]:evaluate()

end

function layer:train()

	if self.clones == nil then self:createClone() end
	assert( #self.clones == 2)

	self.product:train()
	self.tanh:train()
	self.combineSen:train()
	self.predict:train()
	self.clones[1]:train()
	self.clones[2]:train()

end

function layer:createClones()

	self.clones = {self.core}
	self.clones[2] = self.core:clone('weight', 'gradWeight', 'bias', 'gradBias')

end

-- inputs[1] is overall feature, inputs[2] is local feature
function layer:updateOutput(inputs)

	assert( #inputs == 3, 'it is not excepted')

	local imgA = inputs[1]
	local imgB = inputs[2]
	local seq = inputs[3]
	-- seq[torch.eq(seq, 0)] = 1
	-- print(seq)

	if self.clones == nil then self:createClones() end

	local batch_size = imgA:size(1)
	self.inputS = self.combineSen:forward(seq)
	--self.inputS = seq
	--print(self.inputS)
	self.inputA = self.cstAtt_al:forward({imgA, self.inputS})
	self.inputB = self.cstAtt_lc:forward({imgB, self.inputS})
	-- self.inputA = imgA
	-- self.inputB = imgB
	self.inputBpro = self.product:forward(self.inputB)
	self.input_local = self.tanh:forward(self.inputBpro)


	self:_createInitState(batch_size)

	self.state = {[0] = self.init_state}
	self.GRUinputs = {}
	self.output = {}

	local xt
	for i=1,2 do

		if i==1 then xt = self.inputA
		else xt =self.input_local end
		self.GRUinputs[i] = {xt, unpack(self.state[i-1])}

		--print(i)
		--print(self.GRUinputs[i])
		local out = self.clones[i]:forward(self.GRUinputs[i])
		self.state[i] = {}
		table.insert(self.state[i], out)

	end

	local output = self.predict:forward(self.state[2][1])
	self.output = {}
	table.insert(self.output, self.state[2][1])
	table.insert(self.output, output)

	return self.output

end

-- gradOutput is table, gradOutput[1] is vector, gradOutput[2] is p
function layer:updateGradInput(inputs, gradOutput)

	local imgA = inputs[1]
	local imgB = inputs[2]
	local seq = inputs[3]

	local dout = self.predict:backward(self.state[2][1], gradOutput[2])
	dout:add(gradOutput[1])
	local gradB = self.clones[2]:backward(self.GRUinputs[2], dout)
	--print(gradB[1])

	local doutA = gradB[2]
	local gradA = self.clones[1]:backward(self.GRUinputs[1], doutA)
	--print(gradA[1])

	local gradBB = self.tanh:backward(self.inputBpro, gradB[1])
	gradBB = self.product:backward(self.inputB, gradBB)
	gradBB = self.cstAtt_lc:backward({imgB, self.inputS}, gradBB)


	local gradAA = self.cstAtt_al:backward({imgA, self.inputS}, gradA[1])

	local sen = gradBB[2]:clone()
	sen:add(gradAA[2])

	self.combineSen:backward(seq, sen)
 	self.gradInput = {}
	table.insert(self.gradInput, gradAA[1])
	table.insert(self.gradInput, gradBB[1])
	table.insert(self.gradInput, sen)

	return self.gradInput

end


local crit, parent = torch.class('nn.AttentionCriterion', 'nn.Criterion')
function crit:__init(opt)
	parent.__init(self)
	self.seq_length = utils.getopt(opt, 'seq_length')
end

function crit:createTarget(batch_size)

	if self.target == nil then self.target = torch.LongTensor(batch_size):zero()
	elseif self.target:size(1) ~= batch_size then self.target:resize(batch_size):zero()
	else self.target:zero() end

end

function crit:clearTarget()

	self.target:zero()

end

-- input is table, input[1] is NXV
-- input[2] is NXV
-- Dth word is target word
function crit:updateOutput(input, seq)

	-- print(seq)
	-- print(input)
	-- print(seq)
	local D,N,MP1 = seq:size(1), seq:size(2), input:size(2)
	assert( N == input:size(1))
	-- if first time, set target
	if D==1 then self:createTarget(N) end

	self.gradInput:resizeAs(input):zero()

	local loss = 0
	local n = 0

	for b=1,N do

		local target_index = seq[D][b]
		if self.target[b] == 0 and target_index == 0 then
			target_index = MP1
			self.target[b] = 1
		end

		if target_index ~= 0 then
			loss = loss - input[b][target_index]
			self.gradInput[b][target_index] = -1
			n = n + 1
		end

	end
	if n>0 then self.output = loss/n
	else self.output = loss
	end
	if n>0 then self.gradInput:div(n) end
	return self.output

end

function crit:updateGradInput(input, seq)

	return self.gradInput

end
