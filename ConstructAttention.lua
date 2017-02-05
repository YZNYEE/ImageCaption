require'nn'
require'nngraph'
require'torch'

local layer,parent = torch.class('nn.ConstructAttention','nn.module')
local utils = require'misc.utils'

function layer:_init(opt, subject)

	self.encoding_size = utils.getopt(opt, 'encoding_size', 512)
	self.local_img_num = utils.getopt(opt, 'local_img_num', 196)
	-- get_top_num is the num of the local_img gotten, and 0 represent 'all'
	self.get_top_num = utils.getopt(opt, 'get_top_num', 3)
	-- subject: 'local', 'overall'
	self.subject = subject
	self.batch_size = utils.getopt(opt, 'batch_size', nil)
	self.DotProduct = nn.DotProduct()
	self.Tanh = nn.Tanh()
	self.eltwise = nn.CMulTable()
	assert(self.batch_size ~= nil, 'must assign batch_size')

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

function layer:getModuleList()

	return {self.DotProduct, self.Tanh, self.eltwise}

end

function layer:parameters()

	return {}
'
end

-- inputs[1] is img_feature(DX196X512)
-- inputs[2] is sentence_feature(DX512)
function local_attention(inputs)

	local img = inputs[1]
	local sen = inputs[2]
	local size_img = img:size()
	local size_sen = sen:size()
	assert(size_sen[2] == size_img[3], 'dimension is not accordance')
	assert(size_sen[2] == self.encoding_size, 'encdding_size is not equal with sentence')

	self.beattention = torch.FloatTensor(self.batch_size, size_img[2]):zero()


	for i=1,size_img[1] do
		local belta = self.DotProduct:forward({img:sub(1,self.batch_size,i,i), sen})
		self.beattention:sub(1,self.batch_size,i,i):copy(belta)
	end


	self.attention = torch.FloatTensor(self.batch_size, size_img[2]):zero()
	self.attention = self.Tanh:forward(self.beattention)

	self.ind = torch.Tensor(self.batch_size, size_img[2])

	for i=1, self.batch_size do
		local y,ind = torch.sort(self.attention:sub(i,i))
		self.ind[i] = ind
	end

	self.output = torch.FloatTensor(self.batch_size, size_sen[2]):zero()

	local len = self.get_top_num
	if len = 0 then
		len = size_sen[2]
	end

	self.local_att_img = torch.FloatTensor(batch_size, size_img[2], size_img[3])

	for i=1,self.batch_size do
		self.local_att_img[i] = self.eltwise(img[i], torch.expand(self.attention, 1, size_sen[2]))
	end

	for j=1,self.batch_size do
		for i=1, len do
			self.output[j]:add(self.local_att_img:sub(j,j,ind[j][i], ind[j][i]))
		end
	end
	self.output:div(len)
	return self.output

end

function overall_attention(inputs)

	local img = inputs[1]
	local sen = inputs[2]
	local size_img = img:size()
	local size_sen = sen:size()
	assert(size_img[1] == self.batch_size)
	assert(size_sen[1] == self.batch_size)
	assert(size_img[2] == size_sen[2], 'dimension is not accordance')

	self.beattention = self.eltwise:forward({img, sen})
	self.attention = self.Tanh:forward(self.beattention)
	self.output = self.eltwise:forward({img,attention})

	return self.output

end

function overall_grad(inputs, gradoutput)

	local gimg, gsen = self.eltwise:backward({inputs[1], self.attention}, gradoutput)
	local gradatt = self.Tanh:backward(self.beattention, gsen)
	local grad_img, grad_sen = self.eltwise:backward(inputs, gradatt)
	grad_img:add(gimg)
	return {grad_img, grad_sen}

end

-- inputs[1]:DX196X512
-- inputs[2]:DX512
-- gradoutput:DX512
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

	local grad_att = torch.FloatTensor(size_img[1], size_img[3]):zero()
	local grad_sen = torch.FloatTensor(size_sen):zero()

	for j=1,self.batch_size do
		for i=1,len do
			grad_att:sub(j, j, self.ind[j][i], self.ind[j][i]):copy(grad[j])
		end
	end

	-- grad_att:DX196X512
	local gradimg= torch.FloatTensor(size_img):zero()
	local gradsen = torch.FloatTensor(size_sen):zero()
	-- gradimg:DX196X512
	-- gradsen:DX512

	for j=1,self.batch_size do

		local gimg_1, gsen = self.eltwise:backward({img[j], torch.expand(self.attention[j], 1, size_sen[2])}, grad_att[j])
		gsen = torch.sum(gsen, 2)
		gsen = self.Tanh:backward(self.beattention[j], gsen)

		local flag = false
		for i=1,len do
			if gsen[i] == 0 then
				flag = true
			end
			-- if gradient of belta_i is zero, dont extend backward
			if not flag then
				local gimg, gs = self.DotProduct:backward({img:sub(i,i), sen}, gsen[i])
				gradimg:sub(j,j,i,i):copy(gimg)
				gradsen[j]:add(gs)
			end
		end
		gradimg[j]:add(gimg_1)
	end

	return {gradimg, gradsen}

end
