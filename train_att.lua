
require 'torch'
require 'nn'
require 'nngraph'
-- exotic things
require 'loadcaffe'
-- local imports
local utils = require 'misc.utils'
require 'misc.DataLoadernew'
require 'AttentionModel'
local net_utils = require 'misc.net_utils'
local cnn_utils = require 'cnn_utils'
require 'misc.optim_updates'


-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')


cmd:option('-input_h5','coco/data.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','coco/data.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_proto','model/VGG_ILSVRC_16_layers_deploy.prototxt','path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','model/VGG_ILSVRC_16_layers.caffemodel','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-stack_num',1,'the num of stack for attmodel')

cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',16,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: for the Attention Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')

cmd:option('-val_images_use', 5000, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

-- additional
cmd:option('-num_of_local_img',1,'the number of local image feature')
cmd:option('-index_of_feature',3040,'table contains index of images extracted ')
cmd:option('-get_top_num',5,'number of local img chosen')
cmd:option('-direction','forward','the direction of model')

cmd:text()


local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end


-------------------------------------------------------------------------------
-- Create the Data Loader instance
---------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------

local protos = {}

if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local am_modules = protos.am:getModulesList()
  for k,v in pairs(am_modules) do net_utils.unsanitize_gradients(v) end
  opt.seq_length = loader:getSeqLength()
  protos.crit = nn.AttDisCriterion(opt) -- not in checkpoints, create manually
  protos.expand = nn.FeatExpander(opt.seq_per_img)
  protos.expand3 = nn.FeatExpander_3d(opt.seq_per_img)

else
  -- create protos from scratch
  -- intialize language model
  local amOpt = {}
  amOpt.vocab_size = loader:getVocabSize()
  amOpt.encoding_size = opt.encoding_size
  amOpt.rnn_size = opt.rnn_size
  -- lmOpt.num_layers = opt.num_layers
  amOpt.dropout = opt.drop_prob_lm
  amOpt.seq_length = loader:getSeqLength()
  amOpt.batch_size = opt.batch_size * opt.seq_per_img

  amOpt.get_top_num = opt.get_top_num
  --lmOpt.local_img_num = 14*14
  amOpt.num_of_local_img = opt.num_of_local_img
  amOpt.feature_table = {}
  amOpt.stack_num = opt.stack_num

  for i=1,string.len(opt.index_of_feature),2 do
	  local num = string.sub(opt.index_of_feature, i, i+1)
	  table.insert(amOpt.feature_table, num)
  end

  protos.am = nn.AttentionModel(amOpt)
  -- initialize the ConvNet
  local cnn_backend = opt.backend
  if opt.gpuid == -1 then cnn_backend = 'nn' end -- override to nn if gpu is disabled
  local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, cnn_backend)

  protos.cnn = cnn_utils.build_cnn_total(cnn_raw, {encoding_size = opt.input_encoding_size, backend = cnn_backend})

  protos.expand = nn.FeatExpander(opt.seq_per_img)
  protos.expand3 = nn.FeatExpander_3d(opt.seq_per_img)

  -- criterion for the Attention model
  protos.crit = nn.AttDisCriterion(amOpt)
end

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do
		v:cuda()
  end
end

protos.feature_table = {}
for i=1,string.len(opt.index_of_feature),2 do
	  local num = string.sub(opt.index_of_feature, i, i+1)
	  num = tonumber(num)
	  table.insert(protos.feature_table, num)
end
print(protos.feature_table)

local params, grad_params = protos.am:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
--返回元素的数量
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())

local thin_am = protos.am:clone()
thin_am.core:share(protos.am.core, 'weight', 'bias') -- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_am.predict:share(protos.am.predict, 'weight', 'bias')
thin_am.combineSen.lookup_table:share(protos.am.combineSen.lookup_table, 'weight', 'bias')
thin_am.product:share(protos.am.product, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')

net_utils.sanitize_gradients(thin_cnn)
local am_modules = thin_am:getModulesList()
for k,v in pairs(am_modules) do net_utils.sanitize_gradients(v) end

protos.am:createClones()

collectgarbage() -- "yeah, sure why not"
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------

-- to save time
local exseq = torch.LongTensor(protos.am.seq_length + 1, opt.batch_size * opt.seq_per_img)
exseq:sub(1,1):fill(protos.am.vocab_size+1)
local empty = torch.CudaTensor(opt.batch_size * opt.seq_per_img, protos.am.rnn_size):zero()


local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

  protos.cnn:evaluate()
  protos.am:evaluate()

  loader:resetIterator(split) -- rewind iteator back to first datapoint in the split
  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  while true do

    -- fetch a batch of data
    local data = loader:getBatch{batch_size = opt.batch_size, split = split, seq_per_img = opt.seq_per_img}
    data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
    n = n + data.images:size(1)

    -- forward the model to get loss
	local feats = protos.cnn:forward(data.images)
	local expanded_feats = {}
	--print(feats)
	expanded_feats[1] = protos.expand:forward(feats[1], opt.seq_per_img)
	expanded_feats[2] = protos.expand3:forward(feats[2], opt.seq_per_img):transpose(2,3)

	local seq
	if opt.direction == 'forward' then seq = data.labels
	else
		assert(opt.direction == 'backward')
		seq = data.labels_back
	end

	exseq:sub(2,protos.am.seq_length+1):copy(seq)
	local texseq = exseq:t()

	local loss_it = 0
	local flag = true

	for i=1,protos.am.seq_length do

		if i>1 then
			local sum = torch.sum(seq:sub(i-1, i-1))
			if sum == 0 then flag = false end
		end

		-- if value of sum is zero, it exclaim that all seq is terminated.
		if flag then

			local input = {unpack(expanded_feats)}
			table.insert(input, texseq:sub(1,opt.batch_size * opt.seq_per_img,1,i))
			local logprobs = protos.am:forward(input)
			local loss = protos.crit:forward({logprobs[2], i}, seq)
			loss_it = loss_it + loss
			loss_evals = loss_evals + 1

		end

	end

	loss_sum = loss_sum + loss_it

    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
	print(string.format('evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss_it/protos.am.seq_length))

    -- if we wrapped around the split or used up val imgs budget then bail


    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  return loss_sum/loss_evals
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0

local function lossFun()

  protos.cnn:training()
  protos.am:training()

  grad_params:zero()
  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data
  local data = loader:getBatch{batch_size = opt.batch_size, split = 'train', seq_per_img = opt.seq_per_img}
  data.images = net_utils.prepro(data.images, true, opt.gpuid >= 0) -- preprocess in place, do data augmentation
  -- data.images: Nx3x224x224
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img

  -- forward the ConvNet on images (most work happens here)
  local feats = protos.cnn:forward(data.images)
  local expanded_feats = {}
  --print(feats)
  expanded_feats[1] = protos.expand:forward(feats[1], opt.seq_per_img)
  expanded_feats[2] = protos.expand3:forward(feats[2], opt.seq_per_img):transpose(2,3)
  --print(expanded_feats)
  -- we have to expand out image features, once for each sentence
  -- forward the attention model

  local seq
  if opt.direction == 'forward' then seq = data.labels
  else
	assert(opt.direction == 'backward')
	seq = data.labels_back
  end
  exseq:sub(2,protos.am.seq_length+1):copy(seq)
  local texseq = exseq:t()

  local losssum = 0
  local dgrad_cnn
  local flag = true

  for i=1, protos.am.seq_length do

	--print(flag)
	if i>1 and flag then
		local sum = torch.sum(seq:sub(i-1, i-1))
		--print(seq:sub(i-1,i-1))
		if sum == 0 then flag = false end
	end

	if flag then

		local input = {unpack(expanded_feats)}
		table.insert(input, texseq:sub(1,opt.batch_size * opt.seq_per_img,1,i))
		local logprobs = protos.am:forward(input)
		local loss = protos.crit:forward({logprobs[2], i}, seq)
		losssum = losssum + loss
		-----------------------------------------------------------------------------
		-- Backward pass
		-----------------------------------------------------------------------------
		-- backprop criterion
		local dlogprobs = protos.crit:backward({logprobs[2], i}, seq)
		-- backprop language model
		local dgrad= protos.am:backward(input, {empty, dlogprobs})
		if dgrad_cnn == nil then
			dgrad_cnn = {}
			dgrad_cnn[1] = dgrad[1]:clone()
			dgrad_cnn[2] = dgrad[2]:clone()
		else
			dgrad_cnn[1]:add(dgrad[1])
			dgrad_cnn[2]:add(dgrad[2])
		end
    end

 end

  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after and dgrad_cnn then
    local dfeats = {}
	--print(dgrad_cnn)
	dfeats[1] = protos.expand:backward(feats[1], dgrad_cnn[1])
    dfeats[2] = protos.expand3:backward(feats[2], dgrad_cnn[2]:transpose(2,3))
	--print(dfeats)
	local dx = protos.cnn:backward(data.images, dfeats)
  end

 losssum = losssum/protos.am.seq_length
  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  -----------------------------------------------------------------------------
  if opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  -- and lets get out!
  local losses = { total_loss = losssum }
  return losses
end

local loss0
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
while true do

  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f', iter, losses.total_loss))

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters-1) then

    -- evaluate the validation performance
    local val_loss = eval_split('val', {val_images_use = opt.val_images_use})
    print('validation loss: ', val_loss)
    val_loss_history[iter] = val_loss

    local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    --checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    --save these too for CIDEr/METEOR/etc eval
    --checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

    -- write the full model checkpoint as well if we did better than ever
	local current_score = -val_loss
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        print('best_score : '..best_score)
		save_protos.am = thin_am -- these are shared clones, and point to correct param storage
		save_protos.cnn = thin_cnn
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
      end
    end
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end







