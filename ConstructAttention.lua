require'nn'
require'nngraph'
require'torch'
local utils = require'misc.utils'

local layer,parent = torch.class('nn.ConstructAttention','nn.Module')
function layer:__init(opt, subject)

	parent.__init(self)

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
	self.eltwise_out = nn.CMulTable()
	assert(self.batch_size ~= nil, 'must assign batch_size')

end

function layer:updateOutput(inputs)

	-- inputs[1] is img_feature(DX196X512)
	-- inputs[2] is sentence_feature(DX512)

	local function local_attention(inputs)

		local img = inputs[1]
		local sen = inputs[2]
		local size_img = img:size()
		local size_sen = sen:size()

		--print(size_img)
		--print(size_sen)

		assert(size_sen[2] == size_img[3], 'dimension is not accordance')
		assert(size_sen[2] == self.encoding_size, 'encdding_size is not equal with sentence')

		--self.beattention = torch.FloatTensor(self.batch_size, size_img[2]):zero()

		local expandsen = torch.expand(sen:resize(size_sen[1], 1,size_sen[2]), size_img[1], size_img[2], size_img[3])
		sen:resize(size_sen[1], size_sen[2])
		self.beattention = torch.sum(torch.cmul(img, expandsen), 3)
		self.beattention:div(size_img[3])

		if self.beattention ~= self.Tanh._type then
			self.beattention = self.beattention:type(self.Tanh._type)
		end

		self.attention = self.Tanh:forward(self.beattention)

		self.ind = torch.Tensor(self.batch_size, size_img[2])

		for i=1, self.batch_size do
			local y,ind = torch.sort(self.attention:sub(i,i), true)
			self.ind:sub(i,i):copy(ind)
		end

		self.output = torch.FloatTensor(self.batch_size, size_sen[2]):zero():type(self._type)

		local len = self.get_top_num
		if len == 0 then
			len = size_img[2]
		end

		self.local_att_img = torch.FloatTensor(self.batch_size, size_img[2], size_img[3])

		-- self.attention:Dx196
		self.local_att_img = torch.cmul(img, torch.expand(self.attention:resize(size_img[1], size_img[2], 1), size_img[1], size_img[2], size_img[3]))

		for j=1,self.batch_size do
			for i=1, len do
				self.output[j]:add(self.local_att_img:sub(j,j,self.ind[j][i], self.ind[j][i]))
			end
		end
		self.output:div(len)
		return self.output

	end

	local function overall_attention(inputs)

		local img = inputs[1]
		local sen = inputs[2]
		local size_img = img:size()
		local size_sen = sen:size()
		assert(size_img[1] == self.batch_size)
		assert(size_sen[1] == self.batch_size)
		assert(size_img[2] == size_sen[2], 'dimension is not accordance')

		self.beattention = self.eltwise:forward({img, sen})
		self.attention = self.Tanh:forward(self.beattention)
		self.output = self.eltwise_out:forward({img, self.attention})

		return self.output

	end

	assert(self.subject ~= nil, 'dont assign the subject')
	if self.subject == 'overall' then
		self.output = overall_attention(inputs)
	elseif self.subject == 'local' then
		self.output = local_attention(inputs)
	end
	return self.output

end

function layer:updateGradInput(inputs, GradOutput)

	local function overall_grad(inputs, gradoutput)

		local grad= self.eltwise_out:backward({inputs[1], self.attention}, gradoutput)
		local gimg = grad[1]
		local gsen = grad[2]
		local gradatt = self.Tanh:backward(self.beattention, gsen)
		grad = self.eltwise:backward(inputs, gradatt)
		local grad_img = grad[1]
		local grad_sen = grad[2]
		grad_img:add(gimg)
		return {grad_img, grad_sen}
	end

	-- inputs[1]:DX196X512
	-- inputs[2]:DX512
	-- gradoutput:DX512
	local function local_grad(inputs, gradoutput)

		local img = inputs[1]
		local sen = inputs[2]
		local size_img = img:size()
		local size_sen = sen:size()

		assert(size_sen[2] == size_img[3], 'dimension is not accordance')
		assert(size_sen[2] == self.encoding_size, 'encdding_size is not equal with sentence')

		local len = self.get_top_num
		if len == 0 then
			len = size_img[2]
		end

		local grad = gradoutput:div(len)

		local grad_att = torch.FloatTensor(size_img):zero():type(self._type)
		local grad_sen = torch.FloatTensor(size_sen):zero():type(self._type)

		for j=1,self.batch_size do
			for i=1,len do
				grad_att:sub(j, j, self.ind[j][i], self.ind[j][i]):copy(grad[j])
			end
		end

		-- grad_att:DX196X512
		local gradimg= torch.FloatTensor(size_img):zero():type(self._type)
		local gradsen = torch.FloatTensor(self.batch_size, size_img[2]):zero():type(self._type)
		-- gradimg:DX196X512
		-- gradsen:DX196

		gradimg = torch.cmul(grad_att, torch.expand(self.attention:resize(size_img[1], size_img[2], 1), size_img[1], size_img[2], size_img[3]))
		gradsen = torch.sum(torch.cmul(grad_att, img), 3)

		gradsen = self.Tanh:backward(self.beattention, gradsen)
		gradsen:div(size_img[3])
		local gradsentence = torch.FloatTensor(sen:size()):zero():type(self._type)
		local expandgradsen = torch.expand(gradsen:resize(size_img[1], size_img[2], 1), size_img[1], size_img[2], size_img[3])

		local grad_img2 = torch.cmul(expandgradsen, torch.expand(sen:resize(size_img[1], 1, size_img[3]), size_img[1], size_img[2], size_img[3]))
		sen:resize(size_img[1], size_img[3])
		gradsentence = torch.sum(torch.cmul(expandgradsen, img), 2):resize(size_sen[1], size_sen[2])

		gradimg:add(grad_img2)

		return {gradimg, gradsentence}
	end

	assert(self.subject ~= nil, 'dont assign the subject')
	if self.subject == 'overall' then
		self.gradInput = overall_grad(inputs, GradOutput)
	elseif self.subject == 'local' then
		self.gradInput = local_grad(inputs, GradOutput)
	end
	return self.gradInput

end

function layer:getModuleList()

	return {self.DotProduct, self.Tanh, self.eltwise}

end

function layer:parameters()
	return {}
end


