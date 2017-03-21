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
	self.stack_num = utils.getopt(opt, 'stack_num', 1)
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
	self.max = nn.CMaxTable()
	self:_createInitState(1)

end

function layer:createMax()
	if self.max == nil then self.max = nn.CMaxTable() end
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

	return {self.core, self.combineSen.lookup_table, self.predict, self.product, self.max}

end

function layer:evaluate()

	if self.clones == nil or self.predicts == nil then self:createClones() end
	assert( #self.clones == 2)

	self.combineSen.lookup_table:evaluate()
	self.product:evaluate()
	self.tanh:evaluate()
	self.max:evaluate()
	for k,v in pairs(self.predicts) do v:evaluate() end
	for k,v in pairs(self.clones) do v:evaluate() end

end

function layer:train()

	if self.clones == nil or self.predicts == nil then self:createClone() end
	assert( #self.clones == 2)

	self.combineSen.lookup_table:train()
	self.product:train()
	self.tanh:train()
	self.max:train()
	for k,v in pairs(self.predicts) do v:train() end
	for k,v in pairs(self.clones) do v:train() end

end

function layer:createClones()

	self.clones = {self.core}
	self.predicts = {self.predict}
	for i=2,self.stack_num*2 do
		self.clones[i] = self.core:clone('weight','bias','gradWeight','gradBias')
	end
	if self.stack_num > 1 then
		for i=2,self.stack_num do
			self.predicts[i] = self.predict:clone('weight','bias','gradWeight','gradBias')
		end
	end

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
	self.predict_outputs = {}

	local xt
	for i=1,self.stack_num do

		self.GRUinputs[2*i-1] = {self.inputA, unpack(self.state[2*i-2])}
		local out = self.clones[2*i-1]:forward(self.GRUinputs[2*i-1])
		self.state[2*i-1] = {}
		table.insert(self.state[2*i-1], out)

		self.GRUinputs[2*i] = {self.input_local, unpack(self.state[2*i-1])}
		local out = self.clones[2*i]:forward(self.GRUinputs[2*i])
		self.state[2*i] = {}
		table.insert(self.state[2*i], out)

		local predict = self.predicts[i]:forward(self.state[2*i][1])
		table.insert(self.predict_outputs, predict)

	end

	local output = self.max:forward(self.predict_outputs)
	self.output = {}
	table.insert(self.output, self.state[self.stack_num*2][1])
	table.insert(self.output, output)

	return self.output

end

-- gradOutput is table, gradOutput[1] is vector, gradOutput[2] is p
function layer:updateGradInput(inputs, gradOutput)

	local imgA = inputs[1]
	local imgB = inputs[2]
	local seq = inputs[3]

	local dpredict = self.max:backward(self.predict_outputs, gradOutput[2])
	local daddition = gradOutput[1]
	local gradB
	local gradA
	for i=self.stack_num,1,-1 do

		-- compute local img grad
		--print(self.stack_num)
		--print(self.predicts[i].gradWeight)
		local dout = self.predicts[i]:backward(self.state[i*2][1], dpredict[i])
		dout:add(daddition)
		local dgru = self.clones[i*2]:backward(self.GRUinputs[i*2], dout)
		dout = dgru[2]
		if i == self.stack_num then gradB = dgru[1] else gradB:add(dgru[1]) end

		-- compute overall img grad
		dgru = self.clones[i*2-1]:backward(self.GRUinputs[i*2-1],dout)
		daddition = dgru[2]
		if i == self.stack_num then gradA = dgru[1] else gradA:add(dgru[1]) end

	end

	--local dout = self.predict:backward(self.state[2][1], gradOutput[2])
	--dout:add(gradOutput[1])
	--local gradB = self.clones[2]:backward(self.GRUinputs[2], dout)
	--print(gradB[1])

	--local doutA = gradB[2]
	--local gradA = self.clones[1]:backward(self.GRUinputs[1], doutA)


	local gradBB = self.tanh:backward(self.inputBpro, gradB)
	gradBB = self.product:backward(self.inputB, gradBB)
	gradBB = self.cstAtt_lc:backward({imgB, self.inputS}, gradBB)
	local gradAA = self.cstAtt_al:backward({imgA, self.inputS}, gradA)
	local sen = gradBB[2]
	sen:add(gradAA[2])

	self.combineSen:backward(seq, sen)
 	self.gradInput = {}
	table.insert(self.gradInput, gradAA[1])
	table.insert(self.gradInput, gradBB[1])
	table.insert(self.gradInput, sen)

	return self.gradInput

end

function layer:clone(...)

    local f = torch.MemoryFile("rw"):binary()
    f:writeObject(self)
    f:seek(1)
    local clone = f:readObject()
    f:close()

    clone.combineSen = self.combineSen:clone(...)
	clone.predict = self.predict:clone(...)
	clone.product = self.product:clone(...)
	clone.tanh = self.tanh:clone(...)

    return clone

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


local crit_dis, parent = torch.class('nn.AttDisCriterion', 'nn.Criterion')
function crit_dis:__init(opt)
	parent.__init(self)
end

function crit_dis:reset(MP1)

	if self.flag then
		self.flag:zero()
		self.used:zero()
		self.grad:zero()
		self.exscore:zero()
	else
		self.flag = torch.LongTensor(MP1):zero()
		self.used = torch.LongTensor(MP1):zero()
		self.grad = torch.CudaTensor(MP1):zero()
		self.exscore = torch.CudaTensor(MP1):zero()
	end

end

function getmax(a, b)

	if a>b then return a
	else return b
	end

end

function crit_dis:updateOutput(inputs, seq)

	local t = inputs[2]
	local input = inputs[1]
	local D,N,MP1 = seq:size(1), seq:size(2), input:size(2)
	self.gradInput:resizeAs(input):zero()

	local loss = 0
	local n = 0
	for i=1,N do

		self:reset(MP1)

		local tt = t+1
		if tt > D then tt = D end
		for j=t,tt do

			local target_index = seq[j][i]
			if target_index == 0 then target_index = MP1 end
			self.flag[target_index] = 1

		end

		--print(self.flag)

		for j=t,tt do

			--print({i,j})
			local target_index = seq[j][i]
			if target_index == 0 then target_index = MP1 end
			local flag = true
			if self.used[target_index] == 1 then flag = false end

			if flag then

				-- get score of the target
				local score = input[i][target_index]
				self.used[target_index] = 1

				self.exscore:fill(score)
				-- compute every pair i,j score in ith sample
				local sc = 1-(self.exscore - input[i])
				-- eliminate the words in label
				sc[torch.eq(self.flag, 1)] = 0
				-- get loss
				local ls = torch.cmax(sc, 0)
				loss = loss + torch.sum(ls)
				-- compute grad
				-- first set grad as 1
				self.grad:fill(1)
				-- eliminate the words that contribute nothing
				self.grad[torch.eq(ls, 0)] = 0
				-- summarize the num of words whose contribution is larger than 0
				local num = torch.sum(torch.gt(ls, 0))
				-- set target_index as -n
				self.grad[target_index] = 0-num
				self.gradInput[i]:add(self.grad)
				n = n + 1

			end

		end

	end
	self.output = loss/(n*MP1)
	self.gradInput:div(n*MP1)
	return self.output

end

function crit_dis:updateGradInput()
	return self.gradInput
end
