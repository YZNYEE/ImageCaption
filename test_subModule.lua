require 'torch'
require 'CombineSentence'
require 'ConstructAttention'

local gradcheck = require 'misc.gradcheck'
local MLGRU = require 'MultiLayerGRU'

local tests = {}
local tester =  torch.Tester()

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
		local opt = {}
		opt.input_size = 10
		opt.output_size = 12
		opt.rnn_size = 10
		opt.numlayers = 3
		opt.dropout = 0

		opt.encoding_size = 10
		opt.vocab_size = 5
		opt.seq_length = 6
		opt.batch_size = 5

		opt.local_img_num = 6
		opt.get_top_num = 3
		opt.subject = 'local'

		local core = MLGRU.mlgru(opt.input_size, opt.output_size, opt.rnn_size, opt.numlayers, opt.dropout)
		core:type(dtype)

		local comsen = nn.SenInfo(opt)
		comsen:type(dtype)

		local conatt = nn.ConstructAttention(opt, 'local')
		conatt:type(dtype)

		local conatt_overall = nn.ConstructAttention(opt, 'overall')
		conatt_overall:type(dtype)

		-- test mlgru's interface

		local xt = torch.randn(opt.batch_size, 10):type(dtype)
		local h1 = torch.randn(opt.batch_size, 10):type(dtype)
		local h2 = torch.randn(opt.batch_size, 10):type(dtype)
		local h3 = torch.randn(opt.batch_size, 10):type(dtype)
		local img1 = torch.randn(opt.batch_size, 10):type(dtype)
		local img2 = torch.randn(opt.batch_size, 10):type(dtype)
		local img3 = torch.randn(opt.batch_size, 10):type(dtype)
		local inputs = {xt, h1, img1, h2, img2, h3, img3}

		local output = core:forward(inputs)

		tester:assertlt(torch.max(output[4]:view(-1)), 0)
		tester:assertTensorSizeEq(output[4], {opt.batch_size, opt.output_size})

		-- test CombineSentence's interface

		local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size):type(dtype)
		seq = seq:t()
		local output = comsen:forward(seq)
		local w = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
		comsen:backward(seq, w)

		tester:assertTensorSizeEq(output, {opt.batch_size, opt.encoding_size})

		-- test interface of ConstructAttention

		local img_local = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
		local sen = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

		local output = conatt:forward({img_local, sen})

		local w = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

		local gradInput = conatt:backward({img_local, sen}, w)

		tester:assertTensorSizeEq(output, {opt.batch_size, opt.encoding_size})
		tester:assertTensorSizeEq(gradInput[1], {opt.batch_size, opt.local_img_num, opt.encoding_size})
		tester:assertTensorSizeEq(gradInput[2], {opt.batch_size, opt.encoding_size})

			-- test interface of ConstructAttention about 'overall'

		local img_local = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
		local sen = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

		local output = conatt_overall:forward({img_local, sen})

		local w = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

		local gradInput = conatt_overall:backward({img_local, sen}, w)

		tester:assertTensorSizeEq(output, {opt.batch_size, opt.encoding_size})
		tester:assertTensorSizeEq(gradInput[1], {opt.batch_size, opt.encoding_size})
		tester:assertTensorSizeEq(gradInput[2], {opt.batch_size, opt.encoding_size})



	end
	return f
end


local function gradCheck_MLGRU()

	local dtype = 'torch.DoubleTensor'
	local core = MLGRU.mlgru(10, 12, 10, 3, 0)
	core:type(dtype)

	local xt = torch.randn(5, 10):type(dtype)
	local h1 = torch.randn(5, 10):type(dtype)
	local h2 = torch.randn(5, 10):type(dtype)
	local h3 = torch.randn(5, 10):type(dtype)
	local img1 = torch.randn(5, 10):type(dtype)
	local img2 = torch.randn(5, 10):type(dtype)
	local img3 = torch.randn(5, 10):type(dtype)

	local output = core:forward({xt, h1, img1, h2, img2, h3, img3})
	local w = torch.randn(output[4]:size())
	local w1 = torch.randn(5, 10)
	local w2 = torch.randn(5, 10)
	local w3 = torch.randn(5, 10)


	local loss = torch.sum(torch.cmul(output[4], w))
	local loss1 = torch.sum(torch.cmul(output[1], w1))
	local loss2 = torch.sum(torch.cmul(output[2], w2))
	local loss3 = torch.sum(torch.cmul(output[3], w3))

	loss_sum = loss + loss1 + loss2 + loss3

	local gradOutput = {w1, w2, w3, w}
	local gradInput = core:backward(inputs, gradOutput)

	local function f(x)
		local output = core:forward({xt, h1, img1, h2, img2, x, img3})
		local loss = torch.sum(torch.cmul(output[4], w))
		local loss1 = torch.sum(torch.cmul(output[1], w1))
		local loss2 = torch.sum(torch.cmul(output[2], w2))
		local loss3 = torch.sum(torch.cmul(output[3], w3))
		local loss_sum = loss+loss1+loss2+loss3
		return loss_sum
	end

	local gradInput_num = gradcheck.numeric_gradient(f, h3, 1, 1e-6)

	tester:assertTensorEq(gradInput[6], gradInput_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradInput[6], gradInput_num, 1e-8), 1e-4)

end

local function gradCheck_CombineSentence()

	local dtype = 'torch.DoubleTensor'
	local opt = {}
	opt.vocab_size = 5
	opt.encoding_size = 10
	opt.seq_length = 6
	opt.batch_size = 5

	local comsen = nn.SenInfo(opt)
	comsen:type(dtype)

	local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size):type(dtype)
	seq:t()

	seq[{1, {4, 6}}]:fill(0)
	seq[{4, {5, 6}}]:fill(0)
	seq[{{1,5}, 1}]:fill(0)

	local output = comsen:forward(seq)
	local w = torch.randn(opt.encoding_size)
	local loss = torch.sum(torch.cmul(output, w))
	local gradOutput = w

end

local function gradCheck_ConstructAttention()

	local dtype = 'torch.DoubleTensor'
	local opt = {}

	opt.encoding_size = 10
	opt.local_img_num = 6
	opt.get_top_num = 3
	opt.subject = 'local'
	opt.batch_size = 5

	local conatt = nn.ConstructAttention(opt, opt.subject)
	conatt:type(dtype)

	local img_local = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
	local sen = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

	local output = conatt:forward({img_local, sen})
	local w = torch.randn(opt.batch_size, opt.encoding_size)
	output = output:type(w:type())

	local loss = torch.sum(torch.cmul(output, w))
	local gradOutput = w
	local grad = conatt:backward({img_local, sen}, gradOutput)
	local gradimg = grad[1]
	local gradsen = grad[2]

	local function f1(x)
		local output = conatt:forward({x, sen})
		output = output:type(w:type())
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end

	local function f2(x)
		local output = conatt:forward({img_local, x})
		output = output:type(w:type())
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end


	local gradimg_num = gradcheck.numeric_gradient(f1, img_local, 1, 1e-6)

	tester:assertTensorEq(gradimg, gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradimg, gradimg_num, 1e-8), 1e-4)


	local gradsen_num = gradcheck.numeric_gradient(f2, sen, 1, 1e-6)

	tester:assertTensorEq(gradsen, gradsen_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradsen, gradsen_num, 1e-8), 1e-4)

end

local function gradCheck_ConstructAttention_overall()

	local dtype = 'torch.DoubleTensor'
	local opt = {}

	opt.encoding_size = 10
	opt.local_img_num = 6
	opt.get_top_num = 3
	opt.subject = 'overall'
	opt.batch_size = 5

	local conatt = nn.ConstructAttention(opt, opt.subject)
	conatt:type(dtype)

	local img_local = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
	local sen = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

	local output = conatt:forward({img_local, sen})
	local w = torch.randn(opt.batch_size, opt.encoding_size)
	output = output:type(w:type())

	local loss = torch.sum(torch.cmul(output, w))
	local gradOutput = w
	local grad = conatt:backward({img_local, sen}, gradOutput)
	local gradimg = grad[1]
	local gradsen = grad[2]

	local function f1(x)
		local output = conatt:forward({x, sen})
		output = output:type(w:type())
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end

	local function f2(x)
		local output = conatt:forward({img_local, x})
		output = output:type(w:type())
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end


	local gradimg_num = gradcheck.numeric_gradient(f1, img_local, 1, 1e-6)

	tester:assertTensorEq(gradimg, gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradimg, gradimg_num, 1e-8), 1e-4)


	local gradsen_num = gradcheck.numeric_gradient(f2, sen, 1, 1e-6)

	tester:assertTensorEq(gradsen, gradsen_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradsen, gradsen_num, 1e-8), 1e-4)

end

tests.doubleApiForwardTest = forwardApiTestFactory('torch.DoubleTensor')
tests.floatApiForwardTest = forwardApiTestFactory('torch.FloatTensor')
tests.cudaApiForwardTest = forwardApiTestFactory('torch.CudaTensor')
tests.gradCheck_MLGRU = gradCheck_MLGRU
tests.gradCheck_ConstructAttention = gradCheck_ConstructAttention
tests.gradCheck_ConstructAttention_overall = gradCheck_ConstructAttention_overall

tester:add(tests)
tester:run()
