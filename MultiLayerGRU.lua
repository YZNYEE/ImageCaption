require 'nn'
require 'nngraph'
require 'torch'

-- multilayer_GRU inputs={xt, img1, img2, img3, ...}

local MLGRU = {}

function MLGRU.mlgru(input_size, output_size, rnn_size, n, dropout)
  dropout = dropout or 0

  local inputs = {}
  table.insert(inputs, nn.Identity()())
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
	table.insert(inputs, nn.Identity()()) -- img
  end

  assert(#inputs == 2*n+1,'error')

  function new_input_sum(insize, xv, hv, gv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
	local g2h = nn.Linear(insize, rnn_size)(gv)
    return nn.CAddTable()({i2h, h2h, g2h})
  end


  local x, input_size_L, img
  local outputs = {}

  for L = 1,n do
	prev_h = inputs[2*L]
	img = inputs[2*L + 1]
    if L == 1 then x = inputs[1] else x = outputs[L-1] end
	if L > 1 then if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end end
    if L == 1 then input_size_L = input_size else input_size_L = rnn_size end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h, img))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h, img))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
	-- compute information given by img
	local p3 = nn.Linear(input_size_L, rnn_size)(img)

    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2,p3}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    local next_h = nn.CAddTable()({zh, zhm1})
	table.insert(outputs, next_h)
  end

  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'}

  table.insert(outputs, proj)

  return nn.gModule(inputs, outputs)

end

return GRU
