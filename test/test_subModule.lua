require 'torch'
require 'CombineSentence'
requrie 'ConstructAttention'

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

local function gradCheck_MLGRU()

	local dtype = 'torch.DoubleTensor'
	local core = MLGRU.mlgru(10, 12, 10, 3, 0)
	core:type(dtype)

	local xt = torch.randn(10):type(dtype)
	local h1 = torch.randn(10):type(dtype)
	local h2 = torch.randn(10):type(dtype)
	local h3 = torch.randn(10):type(dtype)
	local img1 = torch.randn(10):type(dtype)
	local img2 = torch.randn(10):type(dtype)
	local img3 = torch.randn(10):type(dtype)

	local inputs = {xt, h1, img1, h2, img2, h3, img3}

	local out = core:forward(inputs)
	local w = torch.randn(out:size(1))

	local loss = torch.sum(torch.cmul(output, w))
	local gradOutput = w
	local gradInput = core:backward(inputs, gradOutput)

	local function f(x)
		local output = core:forward({xt, h1, x, h2, img2, h3, img3})
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end

	local gradInput_num = gradcheck.numeric_gradient(f, x, 1, 1e-6)

	tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)

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

	local conatt = nn.ConstructAttention(opt)
	local img_local = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
	local sen = torch.randn(opt.encoding_size):type(dtype)

	local output = conatt:forward({img_local, sen})
	local w = torch.randn(opt.encoding_size)
	local loss = torch.sum(torch.cmul(output, w))
	local gradOutput = w
	local gradimg, gradsen = conatt:backward({img_local, sen}, gradOutput)

	local function f1(x)
		local output = conatt:forward({x, sen})
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end

	local function f2(x)
		local output = conatt:forward({img_local, x})
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end


	local gradimg_num = gradcheck.numeric_gradient(f1, x, 1, 1e-6)

	tester:assertTensorEq(gradimg, gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradimg, gradimg_num, 1e-8), 1e-4)


	local gradsen_num = gradcheck.numeric_gradient(f2, x, 1, 1e-6)

	tester:assertTensorEq(gradsen, gradsen_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradsen, gradsen_num, 1e-8), 1e-4)

end

tests.gradCheck_MLGRU = gradCheck_MLGRU
tests.gradCheck_ConstructAttention = gradCheck_ConstructAttention

tester:add(tests)
tester:run()
