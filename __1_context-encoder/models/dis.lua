-- Discriminator network structure


local nn = require 'nn'


local Discrim = {}

local SBatchNorm = nn.SpatialBatchNormalization
local SConv = nn.SpatialConvolution
local SFullConv = nn.SpatialFullConvolution
local LeakyReLU = nn.LeakyReLU


function Discrim.weights_init(m)
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
function Discrim.create_model(type, opt)
	assert(	type==64 or type==128, 
			'erorr. type argument must \'64\' or \'128\'.')

	local nc = opt.nc
	local nz = opt.nz
	local ndf = opt.ndf
	local model = nn.Sequential()


	if type == 64 then
		-- state size : (ndf) x 64 x 64
		model:add(SConv(nc, ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(LeakyReLU(0.2, true))
		-- state size : (ndf) x 32 x 32
		model:add(SConv(ndf, 2*ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(2*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (2*ndf) x 16 x 16
		model:add(SConv(2*ndf, 4*ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(4*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (4*ndf) x 8 x 8
		model:add(SConv(4*ndf, 8*ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(8*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (8*ndf) x 4 x 4
		model:add(SConv(8*ndf, 1, 4, 4))
		model:add(nn.Sigmoid())
		-- state size : (1) x 1 x 1

    elseif type == 128 then
		-- state size : (nc) x 128 x 128
		model:add(SConv(nc, ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(LeakyReLU(0.2, true))
		-- state size : (ndf) x 64 x 64
		model:add(SConv(ndf, ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(ndf)):add(LeakyReLU(0.2, true))
		-- state size : (ndf) x 32 x 32
		model:add(SConv(ndf, 2*ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(2*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (2*ndf) x 16 x 16
		model:add(SConv(2*ndf, 4*ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(4*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (4*ndf) x 8 x 8
		model:add(SConv(4*ndf, 8*ndf, 4, 4, 2, 2, 1, 1):noBias())
		model:add(SBatchNorm(8*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (8*ndf) x 4 x 4
		model:add(SConv(8*ndf, 1, 4, 4))
		model:add(nn.Sigmoid())
		-- state size : (1) x 1 x 1
	end

	return model
end


return Discrim



