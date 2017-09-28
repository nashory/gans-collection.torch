-- Generator network structure for ALI.


local nn = require 'nn'


local Generator = {}

local SBatchNorm = nn.SpatialBatchNormalization
local SConv = nn.SpatialConvolution
local SFullConv = nn.SpatialFullConvolution
local ReLU = nn.ReLU
local LeakyReLU = nn.LeakyReLU


function Generator.weights_init(m)
	local name = torch.type(m)
	if name:find('Convolution') then
		m.weight:normal(0.0, 0.02)
		m.bias:fill(0)
	elseif name:find('BarchNormalization') then
		if m.weight then m.weight:normal(1.0, 0.02) end
		if m.bais then m.bais:fill(0) end
	end
end

-- Generator input context to noise
function Generator.create_model(type, opt)
	assert(type==64 or type==128 or type == 256, 'erorr. type argument must \'64\' or \'128\' or \'256\'.')

	local nc = opt.nc
	local nh = opt.nh
	local ngf = opt.ngf
	local model = nn.Sequential()
    
    if type == 256 then
		-- input is (nBottleneck) x 1 x 1
		model:add(SFullConv(nh, 8*ngf, 4, 4):noBias())
		model:add(SBatchNorm(8*ngf)):add(nn.ReLU(true))
		-- state size : (8*ngf) x 4 x 4
		model:add(SFullConv(8*ngf, 4*ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(4*ngf)):add(nn.ReLU(true))
		-- state size : (4*ngf) x 8 x 8
		model:add(SFullConv(4*ngf, 4*ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(4*ngf)):add(nn.ReLU(true))
		-- state size : (4*ngf) x 16 x 16
		model:add(SFullConv(4*ngf, 2*ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(2*ngf)):add(nn.ReLU(true))
		-- state size : (2*ngf) x 32 x 32
		model:add(SFullConv(2*ngf, 2*ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(2*ngf)):add(nn.ReLU(true))
		-- state size : (2*ngf) x 64 x 64
		model:add(SFullConv(2*ngf, ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(ngf)):add(nn.ReLU(true))
		-- state size : (ngf) x 128 x 128
		model:add(SFullConv(ngf, nc, 4, 4, 2, 2, 1, 1))
		model:add(nn.Tanh())
		-- state size : (nc) x 256 x 256	

	elseif type == 128 then
		-- input is (nBottleneck) x 1 x 1
		model:add(SFullConv(nh, 8*ngf, 4, 4):noBias())
		model:add(SBatchNorm(8*ngf)):add(nn.ReLU(true))
		-- state size : (8*ngf) x 4 x 4
		model:add(SFullConv(8*ngf, 4*ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(4*ngf)):add(nn.ReLU(true))
		-- state size : (4*ngf) x 8 x 8
		model:add(SFullConv(4*ngf, 2*ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(2*ngf)):add(nn.ReLU(true))
		-- state size : (2*ngf) x 16 x 16
		model:add(SFullConv(2*ngf, ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(ngf)):add(nn.ReLU(true))
		-- state size : (ngf) x 32 x 32
		model:add(SFullConv(ngf, ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(ngf)):add(nn.ReLU(true))
		-- state size : (ngf) x 64 x 64
		model:add(SFullConv(ngf, nc, 4, 4, 2, 2, 1, 1))
		model:add(nn.Tanh())
		-- state size : (nc) x 128 x 128	
	elseif type == 64 then
		-- input is (nh) x 1 x 1
		model:add(SFullConv(nh, 8*ngf, 4, 4):noBias())
		model:add(SBatchNorm(8*ngf)):add(nn.ReLU(true))
		-- state size : (8*ngf) x 4 x 4
		model:add(SFullConv(8*ngf, 4*ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(4*ngf)):add(nn.ReLU(true))
		-- state size : (4*ngf) x 8 x 8
		model:add(SFullConv(4*ngf, 2*ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(2*ngf)):add(nn.ReLU(true))
		-- state size : (2*ngf) x 16 x 16
		model:add(SFullConv(2*ngf, ngf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(ngf)):add(nn.ReLU(true))
		-- state size : (ngf) x 32 x 32
		model:add(SFullConv(ngf, nc, 4, 4, 2, 2, 1, 1))
		model:add(nn.Tanh())
		-- state size : (ngf) x 64 x 64
	end

	return model
end


return Generator



