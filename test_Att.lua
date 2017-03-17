require 'torch'
require 'AttentionModel'

local GRU = require'misc.GRU'
local LSTM = require'misc.LSTM'
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

local function forwardApiTestFactory(dtype)
  if dtype == 'torch.CudaTensor' then
    require 'cutorch'
    require 'cunn'
  end
  local function f()
    -- create LanguageModel instance
    local opt = {}
    opt.vocab_size = 5
	opt.encoding_size = 11
    opt.rnn_size = 8
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 10
	opt.local_img_num = 9
	opt.get_top_num = 6

	local am = nn.AttentionModel(opt)
    local crit = nn.AttentionCriterion(opt)
    am:type(dtype)
    crit:type(dtype)

    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
    seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 6 }] = 0

	local exseq = torch.LongTensor(opt.seq_length+1, opt.batch_size)
	exseq:sub(2,opt.seq_length+1,1, opt.batch_size):copy(seq)
	exseq[1]:fill(opt.vocab_size+1)

    local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
    local img_o = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
	local input = {img_o, img_l1, exseq:sub(1,1):t()}

	local output = am:forward(input)
	-- output is table
	-- print(output)
	assert(#output == 2)

    tester:assertlt(torch.max(output[2]:view(-1)), 0) -- log probs should be <0

    -- the output[1] should be of size (batch_size, vocab_size + 1)
    -- where the +1 is for the special END token appended at the end.
    tester:assertTensorSizeEq(output[1], {opt.batch_size, opt.rnn_size})
	tester:assertTensorSizeEq(output[2], {opt.batch_size, opt.vocab_size+1})

    local loss = crit:forward(output[2], seq:sub(1,1))
    local gradOutput = crit:backward(output[2], seq:sub(1,1))

	-- gradOutput is table
	-- gradOutput[1] should be equal to gradOutput[2]
    tester:assertTensorSizeEq(gradOutput, {opt.batch_size, opt.vocab_size+1})

    -- make sure the pattern of zero gradients is as expected
    --local gradAbs = torch.max(torch.abs(gradOutput), 3):view(opt.seq_length+1, opt.batch_size)
    --local gradZeroMask = torch.eq(gradAbs,0)
    --local expectedGradZeroMask = torch.ByteTensor(opt.seq_length+1,opt.batch_size):zero()
    --expectedGradZeroMask[{ {1}, {} }]:fill(1) -- first time step should be zero grad (img was passed in)
    --expectedGradZeroMask[{ {5,8}, 1 }]:fill(1)
    --expectedGradZeroMask[{ {6,8}, 6 }]:fill(1)
    --print(seq)
	--print(gradOutput)
	--tester:assertTensorEq(gradZeroMask:float(), expectedGradZeroMask:float(), 1e-8)

	-- gradInput is table
	local empty = torch.zeros(opt.batch_size, opt.rnn_size)
    local gradInput = am:backward(input, {empty, gradOutput})
	-- print({gradInput[1]:size(), '111111111111'})
    tester:assertTensorSizeEq(gradInput[2], {opt.batch_size, opt.local_img_num, opt.encoding_size})
	tester:assertTensorSizeEq(gradInput[1], {opt.batch_size, opt.encoding_size})
    tester:asserteq(gradInput[3]:nElement(), 0, 'grad on seq should be empty tensor')

  end
  return f
end

local function attforseq()

	local opt = {}
    opt.vocab_size = 5
	opt.encoding_size = 11
    opt.rnn_size = 8
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 10
	opt.local_img_num = 9
	opt.get_top_num = 6

	local am = nn.AttentionModel(opt)
    local crit = nn.AttentionCriterion(opt)

    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
    seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 6 }] = 0

	local exseq = torch.LongTensor(opt.seq_length+1, opt.batch_size)
	exseq:sub(2,opt.seq_length+1,1, opt.batch_size):copy(seq)
	exseq[1]:fill(opt.vocab_size+1)
	exseq[torch.eq(exseq,0)] = 1

    local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size)
    local img_o = torch.randn(opt.batch_size, opt.encoding_size)

	local exceptedgrad = torch.FloatTensor(opt.batch_size, opt.seq_length, opt.vocab_size+1)
	--print(seq)

	for i=1,opt.seq_length do

		local input = {img_o, img_l1, exseq:sub(i,i):t()}
		local output = am:forward(input)
		local loss = crit:forward(output, seq:sub(i, i))
		local gradOutput = crit:backward(output, seq:sub(i,i))
		--print(i..'~~~~~~~~~~~~~~~~~~~~~~')
		--print(gradOutput[1])
		--print(gradOutput)
		exceptedgrad:sub(1, opt.batch_size, i,i, 1,opt.vocab_size+1):copy(gradOutput[1])
		local gradInput = am:backward(input, gradOutput)

	end

	local gradAbs = torch.max(torch.abs(exceptedgrad), 3):view(opt.batch_size, opt.seq_length)
	local gradZeroMask = torch.eq(gradAbs,0)
    local expectedGradZeroMask = torch.ByteTensor(opt.batch_size, opt.seq_length):zero()
	expectedGradZeroMask[{ 1, {5,7}}]:fill(1)
	expectedGradZeroMask[{ 6, {6,7}}]:fill(1)

	--print('~~~~~~~~~~~~~~~~~~~~')
	--print(gradZeroMask)
	--print(expectedGradZeroMask)
	tester:assertTensorEq(gradZeroMask:float(), expectedGradZeroMask:float(), 1e-8)

end

local function checkgrad_AM()

	local dtype = 'torch.DoubleTensor'
	local opt = {}
    opt.vocab_size = 5
	opt.encoding_size = 4
    opt.rnn_size = 3
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 2
	opt.local_img_num = 2
	opt.get_top_num = 1

	local am = nn.AttentionModel(opt)
    local crit = nn.AttentionCriterion(opt)

    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
    seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 2 }] = 0

	local exseq = torch.LongTensor(opt.seq_length+1, opt.batch_size)
	exseq:sub(2,opt.seq_length+1,1, opt.batch_size):copy(seq)
	exseq[1]:fill(opt.vocab_size+1)
	exseq[torch.eq(exseq,0)] = 1

    local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
    local img_o = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

	for i=1,opt.seq_length do

		print(' testing '..i..'th seq')

		local input = {img_o, img_l1, exseq:sub(1,i):t()}
		local output = am:forward(input)
		local w = torch.randn(opt.batch_size, opt.vocab_size+1):type(dtype)
		local w_other = torch.randn(opt.batch_size, opt.vocab_size+1):type(dtype)

		local loss1 = torch.sum(torch.cmul(w, output[1]))
		local loss2 = torch.sum(torch.cmul(w, output[2]))
		local losssum = loss1 + loss2
		local gradOutput = w
		local gradInput = am:backward(input, {gradOutput, gradOutput})

		local function f(x)

			local output = am:forward({img_o, x, exseq:sub(1,i):t()})
			local loss1 = torch.sum(torch.cmul(w, output[1]))
			local loss2 = torch.sum(torch.cmul(w, output[2]))
			local losssum = loss1 + loss2
			return losssum

		end

		local gradInput_num = gradcheck.numeric_gradient(f, img_l1, 1, 1e-6)
		tester:assertTensorEq(gradInput[2], gradInput_num, 1e-4)
		tester:assertlt(gradcheck.relative_error(gradInput[2], gradInput_num, 1e-8), 1e-4)

		local function f1(x)

			local output = am:forward({x, img_l1, exseq:sub(1,i):t()})
			local loss1 = torch.sum(torch.cmul(w, output[1]))
			local loss2 = torch.sum(torch.cmul(w, output[2]))
			local losssum = loss1 + loss2
			return losssum

		end

		local gradInput_num1 = gradcheck.numeric_gradient(f1, img_o, 1, 1e-6)
		tester:assertTensorEq(gradInput[1], gradInput_num1, 1e-4)
		tester:assertlt(gradcheck.relative_error(gradInput[1], gradInput_num1, 1e-8), 1e-4)

	end


end

local function checkgrad()

	local dtype = 'torch.DoubleTensor'
	local opt = {}
    opt.vocab_size = 10000
	opt.encoding_size = 4
    opt.rnn_size = 3
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 80
	opt.local_img_num = 2
	opt.get_top_num = 1

	local am = nn.AttentionModel(opt)
    local crit = nn.AttDisCriterion(opt)

    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
	seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 2 }] = 0
	-- seq[{{1,2},1}] = 0
	-- seq[{{1,2},2}] = 1


	local exseq = torch.LongTensor(opt.seq_length+1, opt.batch_size)
	exseq:sub(2,opt.seq_length+1,1, opt.batch_size):copy(seq)
	exseq[1]:fill(opt.vocab_size+1)
	exseq[torch.eq(exseq,0)] = 1

	local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)

	--print(exseq)
	--print(seq)

    local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
    local img_o = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
	--local sentence = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

	--local true_target = torch.LongTensor(opt.batch_size):zero()
	for i=1,opt.seq_length do

		print(' testing '..i..'th seq')

		--local input = {img_o, img_l1, exseq:sub(1,i):t()}
		local input = {img_o, img_l1, exseq:sub(1,i):t()}
		local output = am:forward(input)

		-- crit module will change target's value, so save target is must in the gradcheck
		local target = true_target
		local loss = crit:forward({output[2], i}, seq)
		-- true_target = crit.target:clone()

		local gradOutput = crit:backward({output[2], i}, seq)
		-- print(gradOutput[1])

		local gradInput = am:backward(input, {torch.zeros(opt.batch_size, opt.rnn_size), gradOutput})
		-- print(gradInput[1])

		local function f(x)

			--crit.target:copy(target)
			local input = {img_o, x, exseq:sub(1,i):t()}
			local output = am:forward(input)
			local loss = crit:forward({output[2], i}, seq)
			return loss

		end

		local gradInput_num = gradcheck.numeric_gradient(f, img_l1, 1, 1e-6)
		tester:assertTensorEq(gradInput[2], gradInput_num, 1e-4)
		tester:assertlt(gradcheck.relative_error(gradInput[2], gradInput_num, 1e-8), 1e-4)

		local function f1(x)

			--crit.target:copy(target)
			local input = {x, img_l1, exseq:sub(1,i):t()}
			local output = am:forward(input)
			local loss = crit:forward({output[2], i}, seq)
			return loss

		end

		local gradInput_num1 = gradcheck.numeric_gradient(f1, img_o, 1, 1e-6)
		tester:assertTensorEq(gradInput[1], gradInput_num1, 1e-4)
		tester:assertlt(gradcheck.relative_error(gradInput[1], gradInput_num1, 1e-8), 1e-4)

		local function f2(x)

			crit.target:copy(target)
			local input = {img_o, img_l1, x}
			local output = am:forward(input)
			local loss = crit:forward(output[2], seq:sub(1,i))
			return loss

		end

		-- local gradInput_num1 = gradcheck.numeric_gradient(f2, sentence, 1, 1e-6)
		-- tester:assertTensorEq(gradInput[3], gradInput_num1, 1e-4)
		-- tester:assertlt(gradcheck.relative_error(gradInput[3], gradInput_num1, 1e-8), 1e-4)

		--print(gradInput_num1)
		--print(gradInput[1])
	end

end

local function checkgrad_crit()

	local opt = {}
	opt.batch_size = 5
	opt.vocab_size = 6
	opt.seq_length = 8

	local crit_dis = nn.AttDisCriterion()
	local input = torch.randn(opt.batch_size, opt.vocab_size+1)
	local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
	seq[{{3,4},1}]=0
	seq[{{3,6},3}]=0
	local a = 1

	--print(seq)

	local loss = crit_dis:forward({input, a}, seq)
	local gradInput = crit_dis:backward({input, a}, seq)

	--print(input)
	--print(gradInput)

	local function f(x)

		local loss = crit_dis:forward({input, a}, seq)
		return loss

	end

	local gradInput_num = gradcheck.numeric_gradient(f, input, 1, 1e-6)
	tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)


end

local function overfit()

	local dtype = 'torch.DoubleTensor'
	local opt = {}
    opt.vocab_size = 5
	opt.encoding_size = 7
    opt.rnn_size = 24
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 6
	opt.local_img_num = 10
	opt.get_top_num = 1

	local am = nn.AttentionModel(opt)
    local crit = nn.AttDisCriterion(opt)

    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    -- make sure seq can be padded with zeroes and that things work ok
	seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 2 }] = 0

	local img_l1 = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
    local img_o = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

	local exseq = torch.LongTensor(opt.seq_length+1, opt.batch_size)
	exseq:sub(2,opt.seq_length+1,1, opt.batch_size):copy(seq)
	exseq[1]:fill(opt.vocab_size+1)
	exseq[torch.eq(exseq,0)] = 1

	local params, grad_params = am:getParameters()
	print('number of parameters:', params:nElement(), grad_params:nElement())

	local gru_params = 3*(opt.encoding_size + opt.rnn_size)*opt.rnn_size + 3*2*opt.rnn_size
	local gru_linear = (opt.vocab_size + 1) * opt.rnn_size + opt.vocab_size+1
	local table_params = (opt.vocab_size + 1) * opt.encoding_size
	local product_params = opt.encoding_size * opt.encoding_size + opt.encoding_size

	local expected_params = gru_params + gru_linear + table_params + product_params
	print('expected:', expected_params)

	local function lossFun()
		grad_params:zero()

		local losssum = 0
		local j=opt.seq_length
		for i=1,j do

			local output = am:forward{img_o, img_l1, exseq:sub(1,i):t()}
			local loss = crit:forward({output[2], i},seq)
			local gradOutput = crit:backward({output[2], i}, seq)
			am:backward({img_o, img_l1, exseq:sub(1,i):t()}, {torch.zeros(opt.batch_size, opt.rnn_size), gradOutput})
			-- print({loss,loss1,loss2})
			losssum = losssum + loss

		end
		return losssum/j
	end

	local loss
	local grad_cache = grad_params:clone():fill(1e-8)
	print('trying to overfit the language model on toy data:')
	for t=1,30 do
		loss = lossFun()
		-- test that initial loss makes sense
		--if t == 1 then tester:assertlt(math.abs(math.log(opt.vocab_size+1) - loss), 0.1) end
		grad_cache:addcmul(1, grad_params, grad_params)
		params:addcdiv(-1e-1, grad_params, torch.sqrt(grad_cache)) -- adagrad update
		print(string.format('iteration %d/30: loss %f ', t, loss))
	end
  -- holy crap adagrad destroys the loss function!

	tester:assertlt(loss, 0.2)

end

--tests.floatApiForwardTest = forwardApiTestFactory('torch.FloatTensor')
--tests.attforseq = attforseq
--tests.gradcheckAM = checkgrad_AM
--tests.gradcheckGRU = check_GRU
tests.gradcheck = checkgrad
tests.overfit = overfit
tests.gradcheck_dis = checkgrad_crit

tester:add(tests)
tester:run()
