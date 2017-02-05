local utils = require 'misc.utils'
local net_utils = {}

-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn(cnn, opt)
  --utils.getopt(a,b,c) 其中c为默认参数
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
  --nn.Sequential()是容器类，队列型容器
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      --将BGR形式的参数形式转换为RGB格式的参数
	  --Clone参数
	  local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end
	--添加网络层
    cnn_part:add(layer)
  end
  --这时已经得到的是cnn的最后一层
  --如果想要提取前面几层可以在这重写
  --在最后一层添加到encoding_size维度的转换
  cnn_part:add(nn.Linear(4096,encoding_size))
  --添加非线性层
  --这里backend为cunn字符串
  cnn_part:add(backend.ReLU(true))
  return cnn_part
end

--提取batchsize长度的images并且进行预处理
-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
-- VGG-16 网络是写死的网络，224大小是网络初始层固定的大小
function net_utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  --得到图片的高与宽
  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = 224

  -- cropping data augmentation, if needed
  -- 裁剪多余的数据，
  if h > cnn_input_size or w > cnn_input_size then
    local xoff, yoff
    if data_augment then
	  --这与下面不同之处是随机从图片中提取224大小的区域
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end

  -- ship to gpu or convert from byte to float
  --转换数据格式
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end

  -- lazily instantiate vgg_mean
  if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
  end
  --typsAs()是按照imgs的格式重新返回一个tensor
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

  -- subtract vgg mean
  imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

  --这个预处理过程，实际上是VGG-16去中值的过程
  return imgs
  --返回经过处理之后的数据
end

-- layer that expands features out so we can forward multiple sentences per image
-- 创建一个类nn.FeatExpander继承于nn.Module,layerS是对每个图片对应的句子的扩充
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
-- n为FeatExpander中要扩充Feature的数量
function layer:__init(n)
  --self相当于C++中的this
  parent.__init(self)
  self.n = n
end
function layer:updateOutput(input)
  --若layer中n为1，则并不需要扩充
  if self.n == 1 then self.output = input; return self.output end -- act as a noop for efficiency
  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  --将每组数据一一拷贝
  for k=1,input:size(1) do
	--这里的K也可以看作是指定的是第几号数据
    local j = (k-1)*self.n+1
	--值得注意的是这里的数据是成块的，即j与j+self之间都是相同的数据�
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end
function layer:updateGradInput(input, gradOutput)
  --n为1，无操作
  if self.n == 1 then self.gradInput = gradOutput; return self.gradInput end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
	--对每一列进行一个求和操作，即相同数据对应的梯度求和
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

--暂时不懂，先打个问号
function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end

function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
  --listModules将net中的Modules转换成LIST
    moduleList = net:listModules()
  end
  return moduleList
end

--取消梯度
function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

--重新开启梯度
function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
--解码
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
	  --..字符串连接
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  --out为全部的词表
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

return net_utils
