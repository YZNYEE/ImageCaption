require'nn'
require'nngraph'
require'torch'

local layer,parent = torch.class('nn.','nn.module')
local utils = require'misc.utils'

function layer:_init(opt)

	self.encoding_size = utils.getopt(opt, 'encoding_size', 512)
	self.local_img_num = utils.getopt(opt, 'local_img_num', 196)
	-- get_top_num is the num of the local_img gotten, and 0 represent 'all'
	self.get_top_num = utils.getopt(opt, 'get_top_num', 3)
	-- subject: 'local', 'overall'
	self.subject = utils.getopt(opt, 'subject', nil)
	self.DotProduct = nn.DotProduct()
	self.Tanh = nn.Tanh()
	self.eltwise = nn.CMulTable()

end

function layer:updataOutput(inputs)

	assert(self.subject ~= nil, 'dont assign the subject')
	if self.subject == 'overall' then
		self.output = overall_attention(inputs)
	elseif self.subject == 'local' then
		self.output = local_attention(inputs)
	end
	return self.output

end

function layer:updataGradInput(inputs, GradOutput)

	assert(self.subject ~= nil, 'dont assign the subject')
	if self.subject == 'overall' then
		self.gradinput = overall_grad(inputs, GradOutput)
	elseif self.subject == 'local' then
		self.gradinput = local_grad(inputs, GradOutput)
	end
	return self.gradinput

end

function local_attention(inputs)

	local img = inputs[1]
	local sen = inputs[2]
	local size_img = img:size()
	local size_sen = sen:size()
	assert(size_sen[1] == size_img[2], 'dimension is not accordance')
	assert(size_sen[1] == self.encoding_size, 'encdding_size is not equal with sentence')

	self.attention = torch.FloatTensor(size_img[1]):zero()

	for i=1,size_img[1] do
		local belta = self.DotProduct:forward({img:sub(i,i), sen})
		self.attention:sub(i,i):copy(belta)
	end

	self.beattention = self.attention
	self.attention = torch.FloatTensor(size_img[1]):zero()
	self.attention = self.Tanh:forward(self.beattention)

	local y,ind = torch.sort(self.attention)
	self.ind = ind

	self.output = torch.FloatTensor(size_sen[1]):zero()

	local len = self.get_top_num
	if len = 0 then
		len = size_sen[1]
	end

	self.local_att_img = self.eltwise(img, torch.expand(self.attention, 1, size_sen[1]))
	for i=1, len do
		self.output:add(self.local_att_img:sub(ind[i], ind[i]))
	end
	self.output:div(len)
	return self.output

end

function overall_attention(inputs)

	local img = inputs[1]
	local sen = inputs[2]
	local size_img = img:size()
	local size_sen = sen:size()
	assert(size_img[1] == size_sen[1], 'dimension is not accordance')

	self.beattention = self.eltwise:forward({img, sen})
	self.attention = self.Tanh:forward(self.beattention)
	self.output = self.attention

	return self.output

end

function overall_grad(inputs, gradoutput)

	local gradatt = self.Tanh:backward(self.beattention, gradoutput)
	local grad_img, grad_sen = self.eltwise:backward(inputs, gradatt)
	return {grad_img, grad_sen}

end

function local_grad(inputs, gradoutput)

	local img = inputs[1]
	local sen = inputs[2]
	local size_img = img:size()
	local size_sen = sen:size()
	local len = self.get_top_num
	if len = 0 then
		len = size_sen[1]
	end

	local grad = gradoutput:div(len)

	local grad_att = torch.FloatTensor(size_img):zero()
	local grad_sen = torch.FloatTensor(size_sen)

	for i=1,len do
		grad_att:sub(self.ind[i], self.ind[i]):copy(grad)
	end

	local gimg_1, gsen = self.eltwise:backward({img, torch.expand(self.attention, 1, size_sen[1])}, grad_att)
	gsen = torch.sum(gsen, 2)
	gsen = self.Tanh:backward(self.beattention, gsen)
	local gradimg= torch.FloatTensor(size_img):zero()
	local gradsen = torch.FloatTensor(size_sen):zero()

	local flag = false
	for i=1,len do
		if gsen[i] == 0 then
			flag = true
		-- if gradient of belta_i is zero, dont extend backward
		if not flag then
			local gimg, gs = self.DotProduct:backward({img:sub(i,i), sen}, gsen[i])
			gradimg:sub(i,i):copy(gimg)
			gradsen:add(gs)
		end
	end

	gradimg:add(gimg_1)
	return {gradimg, gradsen}

end
