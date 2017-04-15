--[[
Unit tests for the LanguageModel implementation, making sure
that nothing crashes, that we can overfit a small dataset
and that everything gradient checks.
--]]

require 'torch'
require 'LanguageModel_addtiional'

local gradcheck = require 'misc.gradcheck'

local tests = {}
local tester = torch.Tester()

-- validates the size and dimensions of a given
-- tensor a to be size given in table sz
function tester:assertTensorSizeEq(a, sz)
  tester:asserteq(a:nDimension(), #sz)
  for i=1,#sz do
    tester:asserteq(a:size(i), sz[i])
  end
end

-- Test the API of the Language Model
local function forwardApiTestFactory(dtype)
  if dtype == 'torch.CudaTensor' then
    require 'cutorch'
    require 'cunn'
  end
  local function f()
    -- create LanguageModel instance
    local opt = {}
    opt.vocab_size = 5
    opt.input_encoding_size = 11
	opt.encoding_size = 11
    opt.rnn_size = 11
	opt.g_size = 11
    opt.num_layers = 2
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 10

	opt.local_img_num = 9
	opt.get_top_num = 6
	opt.num_of_local_img = 1

	opt.finetune_att = true
	opt.attmodel_path = 0

	local lm = nn.LanguageModel(opt)
    local crit = nn.LanguageModelCriterion()
    lm:type(dtype)
    crit:type(dtype)

    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
    seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 6 }] = 0
    local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.input_encoding_size):type(dtype)
    local img_o = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)
	local input = {img_o, img_l1, seq}
	local output = lm:forward(input)
    tester:assertlt(torch.max(output:view(-1)), 0) -- log probs should be <0

    -- the output should be of size (seq_length + 2, batch_size, vocab_size + 1)
    -- where the +1 is for the special END token appended at the end.
    tester:assertTensorSizeEq(output, {opt.seq_length+2, opt.batch_size, opt.vocab_size+1})

    local loss = crit:forward(output, seq)

    local gradOutput = crit:backward(output, seq)
    tester:assertTensorSizeEq(gradOutput, {opt.seq_length+2, opt.batch_size, opt.vocab_size+1})

    -- make sure the pattern of zero gradients is as expected
    local gradAbs = torch.max(torch.abs(gradOutput), 3):view(opt.seq_length+2, opt.batch_size)
    local gradZeroMask = torch.eq(gradAbs,0)
    local expectedGradZeroMask = torch.ByteTensor(opt.seq_length+2,opt.batch_size):zero()
    expectedGradZeroMask[{ {1}, {} }]:fill(1) -- first time step should be zero grad (img was passed in)
    expectedGradZeroMask[{ {6,9}, 1 }]:fill(1)
    expectedGradZeroMask[{ {7,9}, 6 }]:fill(1)
    --print(seq)
	--print(gradOutput)
	tester:assertTensorEq(gradZeroMask:float(), expectedGradZeroMask:float(), 1e-8)

    local gradInput = lm:backward(input, gradOutput)
	-- print({gradInput[1]:size(), '111111111111'})
    -- tester:assertTensorSizeEq(gradInput[1], {opt.batch_size, opt.local_img_num, opt.input_encoding_size})
	-- tester:assertTensorSizeEq(gradInput[2], {opt.batch_size, opt.input_encoding_size})
    -- tester:asserteq(gradInput[3]:nElement(), 0, 'grad on seq should be empty tensor')

  end
  return f
end

-- test just the language model alone (without the criterion)
local function gradCheckLM()

  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.encoding_size = 4
  opt.g_size = 4
  opt.rnn_size = 4
  opt.num_layers = 1
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6

  opt.local_img_num = 9
  opt.get_top_num = 6
  opt.num_of_local_img = 1

  opt.finetune_att = true

  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  -- seq[{ {4, 7}, 1 }] = 0
  -- seq[{ {5, 7}, 4 }] = 0
  local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.input_encoding_size):type(dtype)
  local img_o = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  -- evaluate the analytic gradient
  local input = {img_o, img_l1, seq}

  local output = lm:forward(input)
  local w = torch.randn(output:size(1), output:size(2), output:size(3))
  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output, w))
  local gradOutput = w
  local gradInput = lm:backward(input, gradOutput)

  -- create a loss function wrapper
  print(gradInput[1])

  local function f1(x)
    local output = lm:forward{x, img_l1, seq}
    local loss = torch.sum(torch.cmul(output, w))
    return loss
  end

  local function f2(x)
    local output = lm:forward{img_o, x, seq}
    local loss = torch.sum(torch.cmul(output, w))
    return loss
  end


  local gradInput_num_l1 = gradcheck.numeric_gradient(f2, img_l1, 1, 1e-6)
  local gradInput_num_o = gradcheck.numeric_gradient(f1, img_o, 1, 1e-6)

  print(gradInput_num_o)

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput[2], gradInput_num_l1, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput[2], gradInput_num_l1, 1e-8), 1e-4)

  -- tester:assertTensorEq(gradInput[2], gradInput_num_l2, 1e-4)
  -- tester:assertlt(gradcheck.relative_error(gradInput[2], gradInput_num_l2, 1e-8), 1e-4)

  tester:assertTensorEq(gradInput[1], gradInput_num_o, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput[1], gradInput_num_o, 1e-8), 1e-4)
end

local function gradCheck()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6

  opt.local_img_num = 9
  opt.get_top_num = 6
  opt.num_of_local_img = 1


  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.input_encoding_size):type(dtype)
  -- local img_l2 = torch.randn(opt.batch_size, opt.local_img_num, opt.input_encoding_size):type(dtype)
  local img_o = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  -- evaluate the analytic gradient
  local input = {img_l1, img_o, seq}
  local output = lm:forward(input)
  local loss = crit:forward(output, seq)
  local gradOutput = crit:backward(output, seq)
  local gradInput = lm:backward(input, gradOutput)

  -- create a loss function wrapper
  local function f1(x)
    local output = lm:forward{x, img_o, seq}
    local loss = crit:forward(output, seq)
    return loss
  end

  local function f2(x)
    local output = lm:forward{img_l1, x,  seq}
    local loss = crit:forward(output, seq)
    return loss
  end

  local function f3(x)
    local output = lm:forward{img_l1, img_l2, x, seq}
    local loss = crit:forward(output, seq)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f1, img_l1, 1, 1e-6)

  tester:assertTensorEq(gradInput[1], gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput[1], gradInput_num, 1e-8), 5e-4)

  -- local gradInput_num = gradcheck.numeric_gradient(f2, img_l2, 1, 1e-6)

  -- tester:assertTensorEq(gradInput[2], gradInput_num, 1e-4)
  -- tester:assertlt(gradcheck.relative_error(gradInput[2], gradInput_num, 1e-8), 5e-4)

  local gradInput_num = gradcheck.numeric_gradient(f2, img_o, 1, 1e-6)

  tester:assertTensorEq(gradInput[2], gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput[2], gradInput_num, 1e-8), 5e-4)
end

local function overfit()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 17
  opt.encoding_size = 17
  opt.g_size = 17
  opt.rnn_size = 17
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6

  opt.local_img_num = 9
  opt.get_top_num = 6
  opt.num_of_local_img = 1

  opt.finetune_att = true
  opt.attmodel_path = 0

  local lm = nn.LanguageModel(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.input_encoding_size):type(dtype)
  local img_o = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local params, grad_params = lm:getParameters()
  print('number of parameters:', params:nElement(), grad_params:nElement())

  local gru_params_1 = 3*(opt.input_encoding_size * 2 + opt.rnn_size )*opt.rnn_size + 3*3*opt.rnn_size
  local output_params = opt.rnn_size * (opt.vocab_size + 1) + opt.vocab_size+1
  local table_params = (opt.vocab_size + 1) * opt.input_encoding_size

  local expected_params = gru_params_1  + output_params + table_params
  print('expected:', expected_params)

  local function lossFun()
    grad_params:zero()
    local output = lm:forward{img_o, img_l1, seq}
    local loss = crit:forward(output, seq)
    local gradOutput = crit:backward(output, seq)
    lm:backward({img_o, img_l1, seq}, gradOutput)
    return loss
  end

  local loss
  local grad_cache = grad_params:clone():fill(1e-8)
  print('trying to overfit the language model on toy data:')
  for t=1,30 do
    loss = lossFun()
    -- test that initial loss makes sense
    if t == 1 then tester:assertlt(math.abs(math.log(opt.vocab_size+1) - loss), 0.1) end
    grad_cache:addcmul(1, grad_params, grad_params)
    params:addcdiv(-1e-1, grad_params, torch.sqrt(grad_cache)) -- adagrad update
    print(string.format('iteration %d/30: loss %f', t, loss))
  end
  -- holy crap adagrad destroys the loss function!

  tester:assertlt(loss, 0.2)
end

-- check that we can call :sample() and that correct-looking things happen
local function sample()
  local dtype = 'torch.DoubleTensor'
  local opt = {}
  opt.vocab_size = 5
  opt.input_encoding_size = 4
  opt.encoding_size = 4
  opt.rnn_size = 4
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6

  opt.local_img_num = 9
  opt.get_top_num = 6
  opt.num_of_local_img = 1

  opt.finetune_att = true
  opt.attmodel_path = 0

  local lm = nn.LanguageModel(opt)
  local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.input_encoding_size):type(dtype)
  local img_o = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)
  local seq = lm:sample({img_o,  img_l1})

  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 1)
  tester:assertle(torch.max(seq), opt.vocab_size+1)
  print('\nsampled sequence:')
  print(seq)
end


-- check that we can call :sample_beam() and that correct-looking things happen
-- these are not very exhaustive tests and basic sanity checks
local function sample_beam()
  local dtype = 'torch.DoubleTensor'
  torch.manualSeed(1)

  local opt = {}
  opt.vocab_size = 10
  opt.input_encoding_size = 4
  opt.encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6

  opt.local_img_num = 9
  opt.get_top_num = 6
  opt.num_of_local_img = 1

  local lm = nn.LanguageModel(opt)

  local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.input_encoding_size):type(dtype)
  -- local img_l2 = torch.randn(opt.batch_size, opt.local_img_num, opt.input_encoding_size):type(dtype)
  local img_o = torch.randn(opt.batch_size, opt.input_encoding_size):type(dtype)

  local seq_vanilla, logprobs_vanilla = lm:sample({img_l1, img_l2, img_o})
  local seq, logprobs = lm:sample({img_l1, img_l2, img_o}, {beam_size = 1})

  -- check some basic I/O, types, etc.
  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 0)
  tester:assertle(torch.max(seq), opt.vocab_size+1)

  -- doing beam search with beam size 1 should return exactly what we had before
  print('')
  print('vanilla sampling:')
  print(seq_vanilla)
  print('beam search sampling with beam size 1:')
  print(seq)
  tester:assertTensorEq(seq_vanilla, seq, 0) -- these are LongTensors, expect exact match
  tester:assertTensorEq(logprobs_vanilla, logprobs, 1e-6) -- logprobs too

  -- doing beam search with higher beam size should yield higher likelihood sequences
  local seq2, logprobs2 = lm:sample({img_l1, img_l2, img_o}, {beam_size = 8})
  local logsum = torch.sum(logprobs, 1)
  local logsum2 = torch.sum(logprobs2, 1)
  print('')
  print('beam search sampling with beam size 1:')
  print(seq)
  print('beam search sampling with beam size 8:')
  print(seq2)
  print('logprobs:')
  print(logsum)
  print(logsum2)

  -- the logprobs should always be >=, since beam_search is better argmax inference
  tester:assert(torch.all(torch.gt(logsum2, logsum)))
end

--tests.doubleApiForwardTest = forwardApiTestFactory('torch.DoubleTensor')
tests.floatApiForwardTest = forwardApiTestFactory('torch.FloatTensor')
--tests.cudaApiForwardTest = forwardApiTestFactory('torch.CudaTensor')
--tests.gradCheck = gradCheck
tests.gradCheckLM = gradCheckLM
tests.overfit = overfit
--tests.sample = sample
-- tests.sample_beam = sample_beam

tester:add(tests)
tester:run()
