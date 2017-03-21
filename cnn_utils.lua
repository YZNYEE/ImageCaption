require 'hdf5'
local utils = require 'misc.utils'
require'misc.DataLoader'
local cnn_utils = {}

function cnn_utils.build_cnn(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)

  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = {}
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    table.insert(cnn_part, layer)
  end

  table.insert(cnn_part, nn.Linear(4096,encoding_size))
  table.insert(cnn_part, backend.ReLU(true))
  assert(#cnn_part == 38+2, 'layers of cnn is wrong')
  return cnn_part
end


function cnn_utils.build_cnn_total(cnn, opt)
	  local layer_num = utils.getopt(opt, 'layer_num', 38)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)

  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_head = nn.Sequential()
  local cnn_tail = nn.Sequential()
  local cnn_concat = nn.ConcatTable()

  for i = 1, layer_num do
    local layer = cnn:get(i)
    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    if i <= 30 then
	  cnn_head:add(layer)
	else
	  cnn_tail:add(layer)
	end

  end

  cnn_tail:add(nn.Linear(4096, encoding_size))
  cnn_tail:add(backend.ReLU(true))

  cnn_concat:add(cnn_tail)
  cnn_concat:add(nn.Identity())
  cnn_head:add(cnn_concat)

  assert(#cnn_part == 38+2, 'layers of cnn is wrong')
  return cnn_head

end

function cnn_utils.cnn_check(cnn_part)

	local img = torch.randn(3, 224, 224)
	local input = img
	local size = {}

	for i=1,#cnn_part do
		local layer = cnn_part[i]
		print(i .. 'th layers:')
		local output = layer:forward(input)
		table.insert(size, output:size())
		print(output:size())
		input = output
	end

	return size

end

function cnn_utils.cnn_translate(givenlayer, givenpath, cnn_part, DataLoader)

	local num = #givenlayer
	local num_path = #givenpath
	assert( num == num_path, 'inconsistant')
	assert( num ~= 0, 'givenlayer is null')
	assert( num <= 40, 'number layer is wrong')

	for i=1, num do
		assert(givenlayer[i] <= 40, 'No of layer is beyond the bound')
	end

	local h5_file = {}

	for i=1,num do

		h5_file[i] = hdf5.open(givenpath[i], 'w')

	end

	for i=1,2 do

		print('Translaing '..((i-1)*16+1)..'th ~ '..(i*16)..'th images')
		local data = DataLoader:getBatch(16)
		local img = data.images

		local input = img
		local index = 1
		local count = 1
		for j=1,#cnn_part do

			local output = cnn_part[j]:forward(input)
			input = output
			if j == givenlayer[index]	then

				index = index + 1
				print('	Writing to '..h5_file[count])
				h5_file[count]:write('/imgs', output)
				count = count + 1

			end

		end

	end

end

function cnn_utils.forward(cnn_part, object, img)

	local num = #object
	assert( num <= 40, 'object is not in bound')
	local input = img

	local index = 1
	local outimg = {}

	for i=1,#cnn_part do

		local output = cnn_part[i]:forward(input)
		if index <= num and i == object[index] then
			local img_out = output:clone()
			local size = img_out:size()
			if img_out:nDimension() == 3 then
				img_out = img_out:resize(size[1], size[2]*size[3]):t()
			elseif img_out:nDimension() == 4 then
				img_out = img_out:resize(size[1], size[2], size[3]*size[4]):transpose(2,3)
			end
			table.insert(outimg, img_out)
			index = index + 1
		end
		input = output

	end
	outimg = MirroPro(outimg)
	return outimg

end

-- a is table
function MirroPro(a)

	local len = #a
	local b = {}
	for i=1,len do
		b[i] = a[len-i+1]
	end
	return b

end

function cnn_utils.expand(t, n)

	local tt = {}
	for i=1,#t do

		local array = t[i]
		local size = array:size()
		size[1] = size[1]*n

		local out = torch.Tensor(size):type(array:type())
		for k=1,array:size(1) do
			local j = (k-1)*n+1
			if array:nDimension() == 2 then
				out[{ {j,j+n-1} }] = array[{ {k,k} }]:expand(n, array:size(2)) -- copy over
			elseif array:nDimension() == 3 then
				out[{ {j,j+n-1} }] = array[{ {k,k} }]:expand(n, array:size(2), array:size(3))
			end
		end
		tt[i] = out

	end
	return tt

end

return cnn_utils
