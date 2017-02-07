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

		self.beattention = torch.FloatTensor(self.batch_size, size_img[2]):zero()


		for i=1,size_img[2] do
			local belta = self.DotProduct:forward({img:sub(1,self.batch_size,i,i), sen})
			self.beattention:sub(1,self.batch_size,i,i):copy(belta)
		end

		if self.beattention ~= self.Tanh._type then
			self.beattention = self.beattention:type(self.Tanh._type)
		end

		self.attention = self.Tanh:forward(self.beattention)

		self.ind = torch.Tensor(self.batch_size, size_img[2])

		for i=1, self.batch_size do
			local y,ind = torch.sort(self.attention:sub(i,i), true)
			self.ind:sub(i,i):copy(ind)
		end

		self.output = torch.FloatTensor(self.batch_size, size_sen[2]):zero()

		local len = self.get_top_num
		if len == 0 then
			len = size_sen[2]
		end

		self.local_att_img = torch.FloatTensor(self.batch_size, size_img[2], size_img[3])

		for i=1,self.batch_size do
			self.local_att_img:sub(i,i):copy(self.eltwise:forward({img[i], torch.expand(self.attention[i]:resize(size_img[2],1), size_img[2], size_img[3])}))
		end

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
		self.output = self.eltwise:forward({img,attention})

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

		local gimg, gsen = self.eltwise:backward({inputs[1], self.attention}, gradoutput)
		local gradatt = self.Tanh:backward(self.beattention, gsen)
		local grad_img, grad_sen = self.eltwise:backward(inputs, gradatt)
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
		local len = self.get_top_num
		if len == 0 then
			len = size_sen[1]
		end

		local grad = gradoutput:div(len)

		local grad_att = torch.FloatTensor(size_img):zero()
		local grad_sen = torch.FloatTensor(size_sen):zero()

		for j=1,self.batch_size do
			for i=1,len do
				grad_att:sub(j, j, self.ind[j][i], self.ind[j][i]):copy(grad[j])
			end
		end

		-- grad_att:DX196X512
		local gradimg= torch.FloatTensor(size_img):zero():type(self._type)
		local gradsen = torch.FloatTensor(self.batch_size, size_img[2]):zero():type(self._type)
		-- gradimg:DX196X512
		-- gradsen:DX512

		for j=1,self.batch_size do

			local grad= self.eltwise:backward({img[j], torch.expand(self.attention[j]:resize(size_img[2],1), size_img[2], size_img[3])}, grad_att[j])
			local gimg = grad[1]
			local gsen = grad[2]
			gsen = torch.sum(gsen, 2)
			-- print({self.beattention[j]:size(1), gsen:size(1)})
			-- print({gradsen:sub(j,j), gsen})
			gradsen:sub(j,j):copy(gsen:resize(1,6))
			gradimg:sub(j,j):copy(gimg)

		end
		gradsen = self.Tanh:backward(self.beattention, gradsen)
		local gradsentence = torch.FloatTensor(sen:size()):zero():type(self._type)

		for j=1,size_img[2] do

			local grad = self.DotProduct:backward({img:sub(1,self.batch_size,j,j):resize(self.batch_size, size_img[3]), sen}, gradsen:sub(1,self.batch_size,j,j):resize(self.batch_size))
			local gimg = grad[1]
			local gsen = grad[2]
			gradimg:sub(1,self.batch_size,j,j):add(gimg)
			gradsentence:add(gsen)

		end

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


