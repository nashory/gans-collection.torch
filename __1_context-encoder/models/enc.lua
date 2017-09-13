-- Encoder network structure.


local nn = require 'nn'


local Encoder = {}

local SBatchNorm = nn.SpatialBatchNormalization
local SConv = nn.SpatialConvolution
local SFullConv = nn.SpatialFullConvolution
local LeakyReLU = nn.LeakyReLU


function Encoder.weights_init(m)
	local name = torch.type(m)
	if name:find('Convolution') then
		m.weight:normal(0.0, 0.02)
		m.bias:fill(0)
	elseif name:find('BarchNormalization') then
		if m.weight then m.weight:normal(1.0, 0.02) end
		if m.bais then m.bais:fill(0) end
	end
end

-- Encode input context to noise
function Encoder.create_model(type, opt)
	assert(type==128 or type==64, 'erorr. type argument must be \'64\' or \'128\'.')

	local nc = opt.nc
	local nz = opt.nz
	local nef = opt.nef
	local model = nn.Sequential()

	if type == 128 then
		-- input is (nc) x 128 x 128
		model:add(SConv(nc, nef, 4, 4, 2, 2, 1, 1))
		model:add(nn.ReLU(true))
		-- state size : (nef) x 64 x 64
		model:add(SConv(nef, nef, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(nef)):add(nn.ReLU(true))
		-- state size : (nef) x 32 x 32
		model:add(SConv(nef, 2*nef, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(2*nef)):add(nn.ReLU(true))
		-- state size : (2*nef) x 16 x 16
		model:add(SConv(2*nef, 4*nef, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(4*nef)):add(nn.ReLU(true))
		-- state size : (4*nef) x 8 x 8
		model:add(SConv(4*nef, 8*nef, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(8*nef)):add(nn.ReLU(true))
		-- state size : (8*nef) x 4 x 4
		model:add(SConv(8*nef, nz, 4, 4))
		-- state size: (nz) x 1 x 1
	elseif type == 64 then
		-- state size : (nef) x 64 x 64
		model:add(SConv(nc, nef, 4, 4, 2, 2, 1, 1))
		model:add(nn.ReLU(true))
		-- state size : (nef) x 32 x 32
		model:add(SConv(nef, 2*nef, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(2*nef)):add(nn.ReLU(true))
		-- state size : (2*nef) x 16 x 16
		model:add(SConv(2*nef, 4*nef, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(4*nef)):add(nn.ReLU(true))
		-- state size : (4*nef) x 8 x 8
		model:add(SConv(4*nef, 8*nef, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(8*nef)):add(nn.ReLU(true))
		-- state size : (8*nef) x 4 x 4
		model:add(SConv(8*nef, nz, 4, 4))
		-- state size: (nz) x 1 x 1
	end

	return model
end


return Encoder



