-- Discriminator network structure for ALI.


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
	local njxf = opt.njxf
	local njzf = opt.njzf
	local model = nn.Sequential()

	if type == 128 then
		-- X-Discriminator.
		local XDIS =  nn.Sequential()
		-- input : (nc) x 128 x 128
		XDIS:add(SConv(nc, ndf, 4, 4, 2, 2, 1, 1))
		XDIS:add(LeakyReLU(0.2, true))
		-- state size : (ndf) x 64 x 64
		XDIS:add(SConv(ndf, ndf, 4, 4, 2, 2, 1, 1):noBias())
		XDIS:add(SBatchNorm(ndf)):add(LeakyReLU(0.2, true))
		-- state size : (ndf) x 32 x 32
		XDIS:add(SConv(ndf, 2*ndf, 4, 4, 2, 2, 1, 1):noBias())
		XDIS:add(SBatchNorm(2*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (2*ndf) x 16 x 16
		XDIS:add(SConv(2*ndf, 4*ndf, 4, 4, 2, 2, 1, 1):noBias())
		XDIS:add(SBatchNorm(4*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (4*ndf) x 8 x 8
		XDIS:add(SConv(4*ndf, 8*ndf, 4, 4, 2, 2, 1, 1):noBias())
		XDIS:add(SBatchNorm(8*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (8*ndf) x 4 x 4
		XDIS:add(SConv(8*ndf, njxf, 4, 4):noBias())
		XDIS:add(SBatchNorm(njxf)):add(LeakyReLU(0.2, true))
		-- state size : (njxf) x 1 x 1

		-- Z-Discriminator.
		local ZDIS = nn.Sequential()
		-- input : (nz) x 1 x 1
		ZDIS:add(SConv(nz, njzf, 1, 1, 1, 1)):add(LeakyReLU(0.2, true))
		-- state size : (njzf) x 1 x 1
		ZDIS:add(SConv(njzf, njzf, 1, 1, 1, 1)):add(LeakyReLU(0.2, true))
		-- state size : (njzf) x 1 x 1

		-- XZ-discriminator(joint)
		local XZDIS = nn.Sequential()
		-- input : (njxf*njxf) x 1 x 1
		XZDIS:add(SConv((njxf+njzf), 2*njxf, 1, 1)):add(LeakyReLU(0.2, true))
		-- state size : (2*njxf) x 1 x 1
		XZDIS:add(SConv(2*njxf, 2*njxf, 1, 1)):add(LeakyReLU(0.2, true))
		-- state size : (2*njxf) x 1 x 1
		XZDIS:add(SConv(2*njxf, 1, 1, 1)):add(nn.Sigmoid()):add(nn.View(-1))
		-- state size : (1)
		
		-- build final model
		local XZprl = nn.ParallelTable()
						:add(XDIS)
						:add(ZDIS)
		model:add(XZprl)
		model:add(nn.JoinTable(2))		-- state size : (2*njxf) x 1 x 1
		model:add(XZDIS)				-- state size : 1

	elseif type == 64 then
		-- X-Discriminator.
		local XDIS =  nn.Sequential()
		-- state size : (nc) x 64 x 64
		XDIS:add(SConv(nc, ndf, 4, 4, 2, 2, 1, 1))
		XDIS:add(LeakyReLU(0.2, true))
		-- state size : (ndf) x 32 x 32
		XDIS:add(SConv(ndf, 2*ndf, 4, 4, 2, 2, 1, 1):noBias())
		XDIS:add(SBatchNorm(2*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (2*ndf) x 16 x 16
		XDIS:add(SConv(2*ndf, 4*ndf, 4, 4, 2, 2, 1, 1):noBias())
		XDIS:add(SBatchNorm(4*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (4*ndf) x 8 x 8
		XDIS:add(SConv(4*ndf, 8*ndf, 4, 4, 2, 2, 1, 1):noBias())
		XDIS:add(SBatchNorm(8*ndf)):add(LeakyReLU(0.2, true))
		-- state size : (8*ndf) x 4 x 4
		XDIS:add(SConv(8*ndf, njxf, 4, 4):noBias())
		XDIS:add(SBatchNorm(njxf)):add(LeakyReLU(0.2, true))
		-- state size : (njxf) x 1 x 1

		-- Z-Discriminator.
		local ZDIS = nn.Sequential()
		-- input : (nz) x 1 x 1
		ZDIS:add(SConv(nz, njzf, 1, 1, 1, 1)):add(LeakyReLU(0.2, true))
		-- state size : (njzf) x 1 x 1
		ZDIS:add(SConv(njzf, njzf, 1, 1, 1, 1)):add(LeakyReLU(0.2, true))
		-- state size : (njzf) x 1 x 1

		-- XZ-discriminator(joint)
		local XZDIS = nn.Sequential()
		-- input : (njxf+njzf) x 1 x 1
		XZDIS:add(SConv((njxf+njzf), 2*njxf, 1, 1)):add(LeakyReLU(0.2, true))
		-- state size : (2*njxf) x 1 x 1
		XZDIS:add(SConv(2*njxf, 2*njxf, 1, 1)):add(LeakyReLU(0.2, true))
		-- state size : (2*njxf) x 1 x 1
		XZDIS:add(SConv(2*njxf, 1, 1, 1)):add(nn.Sigmoid()):add(nn.View(-1))
		-- state size : (1)
		
		-- build final model
		local XZprl = nn.ParallelTable()
						:add(XDIS)
						:add(ZDIS)
		model:add(XZprl)
		model:add(nn.JoinTable(2))		-- state size : (2*njf) x 1 x 1
		model:add(XZDIS)				-- state size : 1
    end
	
    return model
end


return Discrim



