require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local GRU = require 'misc.GRU'
local MLGRU = require 'MultiLayerGRU'
require 'ConstructAttention'
require 'CombineSentence'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.encoding_size = utils.getopt(opt, 'encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_of_local_img = utils.getopt(opt, 'num_of_local_img')
  self.num_layers = 1
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length')
  -- create the core lstm network. note +1 for both the START and END tokens
  self.core = MLGRU.mlgru(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  self:_createInitState(1) -- will be lazily resized later during forward passes

  self.core_img = GRU.gru(self.input_encoding_size, self.input_encoding_size, self.input_encoding_size, 1, dropout)

  self.constructAtt_o = nn.ConstructAttention(opt, 'overall')
  self.constructAtt_ls = nn.ConstructAttention(opt, 'local')

  self.combineSen = nn.SenInfo(opt)

  self:_createInitState_img(1)

end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init

  for h=1,self.num_layers do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end

  end
  self.num_state = #self.init_state

end

function layer:_createInitState_img(batch_size)

	assert(batch_size ~= nil, 'batch size must be provided')
	if not self.init_state_img then self.init_state_img = {} end
	if self.init_state_img[1] then
		if self.init_state_img[1]:size(1) ~= batch_size then
			self.init_state_img[1]:resize(batch_size, self.encoding_size):zero()
		end
	else
		self.init_state_img[1] = torch.zeros(batch_size, self.rnn_size)
	end
	self.num_state_img = 1

end



function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  self.combineSens = {self.combineSen}
  for t=2,self.seq_length+2 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
	self.combineSens[t] = self.combineSen:clone('weight','gradWeight')
  end
end

function layer:createClones_img()

  print('constructing clones inside the core_img')
  self.clones_img = {}
  for t=1,self.seq_length+2 do
	self.clones_img[t] = {}
	for h=1,self.num_of_local_img+1 do
		if t==1 and h==1 then
			self.clones_img[t][h] = self.core_img
		else
			self.clones_img[t][h] = self.core_img:clone('weight', 'bias', 'gradWeight', 'gradbias')
		end
	end
  end

end

function layer:getModulesList()
  return {self.core, self.lookup_table, self.core_img, self.combineSen.lookup_table}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()
  local p3,g3 = self.core_img:parameters()
  local p4,g4 = self.combineSen.lookup_table:parameters()

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

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  if self.clones_img == nil then self:createClones_img() end
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.clones_img) do
	for k1,v1 in pairs(v) do
		v1:training()
	end
  end
  for k,v in pairs(self.lookup_tables) do v:training() end
  for k,v in pairs(self.combineSens) do v.lookup_table:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  if self.clones_img == nil then self:createClones_img() end
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.clones_img) do
	for k1,v1 in pairs(v) do
		v1:evaluate()
	end
  end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
  for k,v in pairs(self.combineSens) do v.lookup_table:evaluate() end
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M,
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(imgs, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- indirection for beam search

  assert( #imgs == self.num_of_local_img+1, 'no accordance')


  local batch_size = imgs[#imgs]:size(1)
  self.batch_size = batch_size
  self:_createInitState(batch_size)
  self:_createInitState_img(batch_size)
  local state = self.init_state
  local state_img = self.init_state_img[1]

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  -- A
  local subseq = torch.LongTensor(self.seq_length + 1, batch_size):zero()
  -- A
  for t=1,self.seq_length+2 do

    local xt, it, sampleLogprobs
    if t == 1 then
	  xt = imgs[self.num_of_local_img + 1]
	elseif t == 2 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      xt = self.lookup_table:forward(it)
    else
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      xt = self.lookup_table:forward(it)
    end

    if t > 2 then
      seq[t-2] = it -- record the samples
      seqLogprobs[t-2] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
	end

	if t > 1 then subseq[t-1] = it end

	local imgFeature = {}

	if t > 1 then
		local combineS = self.combineSen:forward(subseq:sub(1,t-1):t())
		for i=1,self.num_of_local_img do
			imgFeature[i] = self.constructAtt_ls:forward({imgs[i], combineS})
		end
		imgFeature[self.num_of_local_img+1] = self.constructAtt_o:forward({imgs[self.num_of_local_img+1], combineS})
		-- assert( #state == 3,'num_layers is not accordance with expectation')
	else
		for i=1,self.num_of_local_img do
			imgFeature[i] = torch.FloatTensor(self.batch_size, self.input_encoding_size):zero():type(self._type)
		end
		imgFeature[self.num_of_local_img+1] = imgs[self.num_of_local_img+1]
	end

	for i=1, self.num_of_local_img+1 do
		inputs = {imgFeature[i]}
		table.insert(inputs, state_img)
		local output = self.core_img:forward(inputs)
		state_img = output
	end

	-- assert( #state == 3,'num_layers is not accordance with expectation')

    inputs = {xt}
	table.insert(inputs, state[1])
	table.insert(inputs, state_img)

    local out = self.core:forward(inputs)
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end

  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
Implements beam search. Really tricky indexing stuff going on inside.
Not 100% sure it's correct, and hard to fully unit test to satisfaction, but
it seems to work, doesn't crash, gives expected looking outputs, and seems to
improve performance, so I am declaring this correct.
]]--
function layer:sample_beam(imgs, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size, feat_dim = imgs[3]:size(1), imgs[3]:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity

  self.constructAtt_l1:changeBatchsize(beam_size)
  self.constructAtt_l2:changeBatchsize(beam_size)
  self.constructAtt_o:changeBatchsize(beam_size)
  self.combineSen:changeBatchsize(beam_size)

  for k=1,batch_size do

    -- create initial states for all beams
    self:_createInitState(beam_size)
    local state = self.init_state

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    local subseq = torch.LongTensor(self.seq_length+1, beam_size):zero()

	local l_1_k = l_1[{{k,k}}]:expand(beam_size, l_1:size(2), l_1:size(3))
	local l_2_k = l_2[{{k,k}}]:expand(beam_size, l_2:size(2), l_2:size(3))
	local overall_k = overall[{{k,k}}]:expand(beam_size, feat_dim)

	for t=1,self.seq_length+1 do

      local xt, it, sampleLogprobs
      local new_state
      if t == 1 then
        -- feed in the start tokens
        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        xt = self.lookup_table:forward(it)
      else
        --[[
          perform a beam merge. that is,
          for every previous beam we now many new possibilities to branch out
          we need to resort our beams to maintain the loop invariant of keeping
          the top beam_size most likely sequences.
        ]]--
        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 2 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 2 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 2 then
            beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+1 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(),
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end

        -- encode as vectors
        it = beam_seq[t-1]
        xt = self.lookup_table:forward(it)
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

	  subseq[t] = it

	  local imgFeature = {}

	  if t > 1 then
		local combineS = self.combineSen:forward(subseq:sub(1,t-1):t())
		imgFeature[1] = self.constructAtt_l1:forward({l_1_k, combineS})
		imgFeature[2] = self.constructAtt_l2:forward({l_2_k, combineS})
		imgFeature[3] = self.constructAtt_o:forward({overall_k, combineS})
		assert( #state == 3,'num_layers is not accordance with expectation')
	  else
		imgFeature[1] = torch.FloatTensor(beam_size, self.input_encoding_size):zero():type(self._type)
		imgFeature[2] = torch.FloatTensor(beam_size, self.input_encoding_size):zero():type(self._type)
		imgFeature[3] = overall_k
	  end

	  assert( #state == 3,'num_layers is not accordance with expectation')

	  inputs = {xt}
	  for i=1,3 do
		table.insert(inputs, state[i])
		table.insert(inputs, imgFeature[i])
	  end

      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code)
2. torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length

returns a (D+1)xNx(M+1) Tensor giving (normalized) log probabilities for the
next token at every iteration of the LSTM (+2 because +1 for first dummy
img forward, and another +1 because of START/END tokens shift)
--]]
function layer:updateOutput(input)

  local seq = input[#input]
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass
  if self.clones_img == nil then self:createClones_img() end

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1)

  self:_createInitState(batch_size)
  self:_createInitState_img(batch_size)

  self.state = {[0] = self.init_state}
  self.state_img = {}
  self.out_img = {}
  self.inputs = {}
  self.inputs_img = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency

  self.subseq = torch.LongTensor(self.seq_length+1, batch_size)
  self.subseq:sub(2,self.seq_length+1, 1,batch_size):copy(seq)
  self.subseq:sub(1,1):fill(self.vocab_size+1)
  self.combineS = {}
  self.batch_size = batch_size

  for t=1,self.seq_length+2 do

    local can_skip = false
    local xt
	if t == 1 then
	  xt = input[self.num_of_local_img+1]
    elseif t == 1 then
      -- feed in the start tokens
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors)
    else
      -- feed in the rest of the sequence...
      local it = seq[t-2]:clone()
      if torch.sum(it) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true
      end
      --[[
        seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
        that won't make lookup_table crash with an error.
        token #1 will do, arbitrarily. This will be ignored anyway
        because we will carefully set the loss to zero at these places
        in the criterion, so computation based on this value will be noop for the optimization.
      --]]
      it[torch.eq(it,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it)
      end
    end

	if not can_skip then
      -- construct the inputs

	  local imgFeature = {}

	  if t > 1 then
		local comseq = self.subseq:sub(1,t-1):t()
		comseq[torch.eq(comseq, 0)] = 1
		self.combineS[t] = self.combineSens[t]:forward(comseq)
		for i=1,self.num_of_local_img do
			-- print({input[i], self.combineS[t]})
			imgFeature[i] = self.constructAtt_ls:forward({input[i], self.combineS[t]})
		end
		imgFeature[self.num_of_local_img+1] = self.constructAtt_o:forward({input[self.num_of_local_img+1], self.combineS[t]})
		-- assert( #state == 3,'num_layers is not accordance with expectation')
	  else
		for i=1,self.num_of_local_img do
			imgFeature[i] = torch.FloatTensor(self.batch_size, self.input_encoding_size):zero():type(self._type)
		end
		imgFeature[self.num_of_local_img+1] = input[self.num_of_local_img+1]
	  end

	  self.state_img[t] = {[0] = self.init_state_img[1]}
	  self.inputs_img[t] = {}

	  for i=1,self.num_of_local_img+1 do
		self.inputs_img[t][i] = {imgFeature[i]}
		table.insert(self.inputs_img[t][i], self.state_img[t][i-1])
		-- rint({i,self.inputs_img[t][i]})
		local out = self.clones_img[t][i]:forward(self.inputs_img[t][i])
		self.state_img[t][i] = out
	  end

	  self.inputs[t] = {xt}
	  table.insert(self.inputs[t], self.state[t-1][1])
	  table.insert(self.inputs[t], self.state_img[t][self.num_of_local_img+1])


	  local out = self.clones[t]:forward(self.inputs[t])
      -- process the outputs
      self.output[t] = out[self.num_state+1] -- last element is the output vector
      self.state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      self.tmax = t
    end
  end

  return self.output
end

--[[
gradOutput is an (D+1)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
  local dimgs -- grad on input images

  -- go backwards and lets compute gradients
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  local dimg_local1, dimg_local2, doverall
  local dgls = {}
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    -- split the gradient to xt and to state
    -- assert( #dinputs == 7)
	local dxt = dinputs[1] -- first element is the input vector
    dstate[t-1] = {} -- copy over rest to state grad

	assert(self.num_layers == 1)
	for k=1,self.num_layers do
		table.insert(dstate[t-1], dinputs[k*2])
	end

    -- continue backprop of xt
	local it = self.lookup_tables_inputs[t]
	if t>1 then self.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table
	else dgls[self.num_of_local_img+1]:add(dxt)

	-- print({self.tmax, t})

	local dsen = torch.FloatTensor(self.batch_size, self.input_encoding_size):zero():type(self._type)

	local dimg_feature = {}

	local dimg = dinputs[3]
	for i=self.num_of_local_img+1,1,-1 do
		local dgrad = self.clones_img[t][i]:backward(self.inputs_img[t][i], dimg)
		dimg_feature[i] = dgrad[1]
		dimg = dgrad[2]
	end

	local dgl = {}

	-- print(dinputs[1])

	if t > 1 then
		for i=1,self.num_of_local_img do
			local dg = self.constructAtt_ls:backward({input[i], self.combineS[t]}, dimg_feature[i])
			dgl[i] = dg[1]
			dsen:add(dg[2])
		end
		local dg = self.constructAtt_o:backward({input[self.num_of_local_img+1], self.combineS[t]}, dimg_feature[self.num_of_local_img+1])
		dgl[self.num_of_local_img+1] = dg[1]
		dsen:add(dg[2])
		self.combineSens[t]:backward(self.subseq:sub(1,t):t(), dsen)
	end


	if t == self.tmax then
		for i=1,self.num_of_local_img+1 do dgls[i] = dgl[i]:clone() end
	elseif t > 1 then
		for i=1,self.num_of_local_img+1 do dgls[i]:add(dgl[i]) end
	else
		dgls[self.num_of_local_img+1]:add(dimg_feature[self.num_of_local_img+1])
	end

  end

  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {}

  for i=1,self.num_of_local_img+1 do table.insert(self.gradInput, dgls[i]) end
  table.insert(self.gradInput, torch.Tensor())
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input is a Tensor of size (D+1)xNx(M+1)
seq is a LongTensor of size DxN. The way we infer the target
in this criterion is as follows:
- at first time step the output is ignored (loss = 0). It's the image tick
- the label sequence "seq" is shifted by one to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-1, 'input Tensor should be 1 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=1,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)

      -- fetch the index of the next token in the sequence
      local target_index
      if t > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t,b}] -- t is correct, since at t=1 START::::token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
	  if target_index == 0 and first_time then
		target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p)
        self.gradInput[{ t,b,target_index }] = -1
        n = n + 1
      end

    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  -- print(self.gradInput)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
