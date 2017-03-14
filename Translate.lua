require 'torch'
require 'nn'
require 'nngraph'
require 'loadcaffe'
local utils = require 'misc.utils'
require 'misc.DataLoader'
local net_utils = require 'misc.net_utils'
local cnn_utils = require 'cnn_utils'

local cnn_proto = 'model/VGG_ILSVRC_16_layers_deploy.prototxt'
local cnn_model = 'model/VGG_ILSVRC_16_layers.caffemodel'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(123)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

local givenlayer = {30, 40}
local givenpath = {'coco/cocotalk_30.h5', 'coco/cocotalk_40.h5'}

local cnn_backend = opt.backend
if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
local cnn_raw = loadcaffe.load(cnn_proto, cnn_model, cnn_backend)
local cnn_part = cnn_utils.build_cnn(cnn_raw, {encoding_size = 512, backend = cnn_backend})

if opt.gpuid >= 0 then
	for k,v in pairs(cnn_part) do
		v:cuda()
	end
end

cnn_utils.cnn_translate(givenlayer, givenpath, cnn_part)
print('Translation has done')

