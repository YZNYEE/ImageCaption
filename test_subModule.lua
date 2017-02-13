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

function abssum(a,b)
	return torch.sum(torch.abs(a-b))
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
	local w1 = torch.randn(5, 10):zero()
	local w2 = torch.randn(5, 10):zero()
	local w3 = torch.randn(5, 10):zero()


	local loss = torch.sum(torch.cmul(output[4], w))
	local loss1 = torch.sum(torch.cmul(output[1], w1))
	local loss2 = torch.sum(torch.cmul(output[2], w2))
	local loss3 = torch.sum(torch.cmul(output[3], w3))

	loss_sum = loss + loss1 + loss2 + loss3

	local gradOutput = {w1, w2, w3, w}
	local gradInput = core:backward(inputs, gradOutput)

	local function f(x)
		local output = core:forward({xt, h1, x, h2, img2, h3, img3})
		local loss = torch.sum(torch.cmul(output[4], w))
		local loss1 = torch.sum(torch.cmul(output[1], w1))
		local loss2 = torch.sum(torch.cmul(output[2], w2))
		local loss3 = torch.sum(torch.cmul(output[3], w3))
		local loss_sum = loss+loss1+loss2+loss3
		return loss_sum
	end

	local gradInput_num = gradcheck.numeric_gradient(f, img1, 1, 1e-6)

	tester:assertTensorEq(gradInput[3], gradInput_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradInput[3], gradInput_num, 1e-8), 1e-4)

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

	local img_local = torch.randn(opt.batch_size, opt.local_img_num + 1, opt.encoding_size):type(dtype)
	local sen = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

	local output = conatt:forward({img_local, sen})
	-- print(output)
	local w = torch.randn(opt.batch_size, opt.encoding_size)
	output = output:type(w:type())
	-- print(output)
	local loss = torch.sum(torch.cmul(output, w))
	local gradOutput = w
	-- print(w)
	local grad = conatt:backward({img_local, sen}, gradOutput)
	local gradimg = grad[1]
	-- print(gradimg)
	local gradsen = grad[2]
	-- print(w)
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
	-- print(gradimg)
	tester:assertTensorEq(gradimg, gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradimg, gradimg_num, 1e-8), 1e-4)


	local gradsen_num = gradcheck.numeric_gradient(f2, sen, 1, 1e-6)

	tester:assertTensorEq(gradsen, gradsen_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradsen, gradsen_num, 1e-8), 1e-4)

end

local function gradCheck_Multi_ConstructAttention()

	local dtype = 'torch.DoubleTensor'
	local opt = {}

	opt.encoding_size = 10
	opt.local_img_num = 6
	opt.get_top_num = 3
	opt.subject = 'local'
	opt.batch_size = 5

	local conatt = nn.ConstructAttention(opt, opt.subject)
	local conatt_2 = nn.ConstructAttention(opt, opt.subject)
	conatt:type(dtype)

	local img_local = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
	local sen = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
	local sen_second = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)


	local output = conatt:forward({img_local, sen}):clone()
	local output_second = conatt_2:forward({img_local, sen_second}):clone()

	local w = torch.randn(opt.batch_size, opt.encoding_size)
	output = output:type(w:type())
	output_sencond = output_second:type(w:type())

	local loss = torch.sum(torch.cmul(output, w))
	local loss_second = torch.sum(torch.cmul(output_second, w))
	local loss_sum = loss + loss_second
	local gradOutput = w

	local grad = conatt:backward({img_local, sen}, gradOutput)
	local grad1 = grad[1]:clone()

	local grad_second = conatt_2:backward({img_local, sen_second}, gradOutput)
	local grad2 = grad_second[1]:clone()

	local gradimg = grad1:clone()
	gradimg:add(grad2)

	-- local gradimg = grad[1]
	-- local gradsen = grad[2]

	local function f_multi(x)
		local output = conatt:forward({x, sen}):clone()
		local output_second = conatt:forward({x, sen_second}):clone()
		output = output:type(w:type())
		output_second = output_second:type(w:type())
		local loss = torch.sum(torch.cmul(output, w))
		local loss_second = torch.sum(torch.cmul(output_second, w))
		return loss + loss_second
	end

	local function f1(x)
		local output = conatt:forward({x, sen}):clone()
		output = output:type(w:type())
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end

	local function f2(x)
		local output = conatt:forward({x, sen_second}):clone()
		output = output:type(w:type())
		local loss = torch.sum(torch.cmul(output, w))
		return loss
	end


	local gradimg_num = gradcheck.numeric_gradient(f_multi, img_local, 1, 1e-6)
	tester:assertTensorEq(gradimg, gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gradimg, gradimg_num, 1e-8), 1e-4)

	local gradimg_num = gradcheck.numeric_gradient(f2, img_local, 1, 1e-6)
	tester:assertTensorEq(grad2, gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(grad2, gradimg_num, 1e-8), 1e-4)

	local gradimg_num = gradcheck.numeric_gradient(f1, img_local, 1, 1e-6)
	tester:assertTensorEq(grad1, gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(grad1, gradimg_num, 1e-8), 1e-4)


end

local function gradCheck_ConstructAttention_Combine_MLGRU()

	local dtype = 'torch.DoubleTensor'
	local opt = {}

	opt.encoding_size = 10
	opt.local_img_num = 6
	opt.get_top_num = 4
	opt.subject = 'local'
	opt.batch_size = 5

	local conatt_1 = nn.ConstructAttention(opt, opt.subject)
	local conatt_2 = nn.ConstructAttention(opt, opt.subject)
	local conatt_3 = nn.ConstructAttention(opt, 'overall')
	conatt_1:type(dtype)
	conatt_2:type(dtype)
	conatt_3:type(dtype)

	local img_local_1 = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
	local img_local_2 = torch.randn(opt.batch_size, opt.local_img_num, opt.encoding_size):type(dtype)
	local img_overall = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
	local sen = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

	local output_1 = conatt_1:forward({img_local_1, sen})
	local output_2 = conatt_2:forward({img_local_2, sen})
	local output_3 = conatt_3:forward({img_overall, sen})


	local xt = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
	local h1 = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
	local h2 = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)
	local h3 = torch.randn(opt.batch_size, opt.encoding_size):type(dtype)

	local core = MLGRU.mlgru(opt.encoding_size, 12, opt.encoding_size, 3, 0)
	core:type(dtype)
	local output = core:forward({xt, h1, output_1, h2, output_2, h3, output_3})
	local w = torch.randn(opt.batch_size, 12):type(dtype)
	output[4] = output[4]:type(w:type())
	local dh1 = torch.randn(opt.batch_size, opt.encoding_size):zero():type(dtype)
	local dh2 = torch.randn(opt.batch_size, opt.encoding_size):zero():type(dtype)
	local dh3 = torch.randn(opt.batch_size, opt.encoding_size):zero():type(dtype)

	local loss = torch.sum(torch.cmul(output[4], w))
	local gradOutput = w

	local grad = core:backward({xt, h1, output_1, h2, output_2, h3, output_3}, {dh1, dh2, dh3,gradOutput})
	-- print(grad[3])
	local grad_l1 = grad[3]

	local grad_l2 = grad[5]
	local grad_overall = grad[7]

	local gl1 = conatt_1:backward({img_local_1, sen}, grad_l1)
	local gl2 = conatt_2:backward({img_local_2, sen}, grad_l2)
	local go = conatt_3:backward({img_overall, sen}, grad_overall)
	-- print(gl1[1])
	-- print(grad_l1)
	local function f(x)
		local out = core:forward({xt, h1, x, h2, output_2, h3, output_3})
		local loss = torch.sum(torch.cmul(out[4], w))
		return loss
	end

	local function f1(x)
		local output = conatt_1:forward({x, sen})
		local out = core:forward({xt, h1, output, h2, output_2, h3, output_3})
		local loss = torch.sum(torch.cmul(out[4], w))
		return loss
	end

	local function f2(x)
		local output = conatt_2:forward({x, sen})
		local out = core:forward({xt, h1, output_1, h2, output, h3, output_3})
		local loss = torch.sum(torch.cmul(out[4], w))
		return loss
	end

	local function f3(x)
		local output = conatt_3:forward({x, sen})
		local out = core:forward({xt, h1, output_1, h2, output_2, h3, output})
		local loss = torch.sum(torch.cmul(out[4], w))
		return loss
	end


	-- print(output_1)
	local gradimg_num = gradcheck.numeric_gradient(f, output_1, 1, 1e-6)
	-- print(grad_l1)
	tester:assertTensorEq(grad_l1, gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(grad_l1, gradimg_num, 1e-8), 1e-4)

	local gradimg_num = gradcheck.numeric_gradient(f1, img_local_1, 1, 1e-6)

	tester:assertTensorEq(gl1[1], gradimg_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gl1[1], gradimg_num, 1e-8), 1e-4)
	--print(gl1[1])
	--print(gradimg_num)

	local gradsen_num = gradcheck.numeric_gradient(f2, img_local_2, 1, 1e-6)

	tester:assertTensorEq(gl2[1], gradsen_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(gl2[1], gradsen_num, 1e-8), 1e-4)

	local gradsen_num = gradcheck.numeric_gradient(f3, img_overall, 1, 1e-6)

	tester:assertTensorEq(go[1], gradsen_num, 1e-4)
	tester:assertlt(gradcheck.relative_error(go[1], gradsen_num, 1e-8), 1e-4)

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
	-- print(output)
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
tests.gradCheck_Multi_ConstructAttention = gradCheck_Multi_ConstructAttention
tests.gradCheck_ConstructAttention_overall = gradCheck_ConstructAttention_overall
tests.gradCheck_ConstructAttention_Combine_MLGRU = gradCheck_ConstructAttention_Combine_MLGRU

tester:add(tests)
tester:run()
