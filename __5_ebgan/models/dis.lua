-- Generator network structure for ALI.


local nn = require 'nn'
require 'cudnn'


local Generator = {}

local SBatchNorm = cudnn.SpatialBatchNormalization
local SConv = cudnn.SpatialConvolution
local SFullConv = cudnn.SpatialFullConvolution


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
	assert(type==64 or type==128, 'erorr. type argument must \'64\' or \'128\'.')

	local nc = opt.nc
	local nz = opt.nh
	local ngf = opt.ngf
    local nef = opt.ngf
	local model = nn.Sequential()
    
	if type == 128 then
        -- encoder.
        -- state size : (3) x 128 x 128 
        local enc = nn.Sequential()
        enc:add(SConv(3, nef, 4, 4, 2, 2, 1, 1))
        enc:add(nn.ReLU(true))
        -- state size : (nef) x 64 x 64
        enc:add(SConv(nef, nef, 4, 4, 2, 2, 1, 1))
        enc:add(SBatchNorm(nef)):add(nn.ReLU(true))
        -- state size : (nef) x 32 x 32
        enc:add(SConv(nef, 2*nef, 4, 4, 2, 2, 1, 1))
        enc:add(SBatchNorm(2*nef)):add(nn.ReLU(true))
        -- state size : (2*nef) x 16 x 16
        enc:add(SConv(2*nef, 4*nef, 4, 4, 2, 2, 1, 1))
        enc:add(SBatchNorm(4*nef)):add(nn.ReLU(true))
        -- state size : (4*nef) x 8 x 8
        enc:add(SConv(4*nef, 8*nef, 4, 4, 2, 2, 1, 1))
        enc:add(SBatchNorm(8*nef)):add(nn.ReLU(true))
        -- state size : (8*nef) x 4 x 4
        enc:add(SConv(8*nef, nz, 4, 4))
        -- state size : (nz) x 1 x 1

        -- decoder.
		-- state size : (nBottleneck) x 1 x 1
		local dec = nn.Sequential()
        dec:add(SFullConv(nz, 8*ngf, 4, 4):noBias())
		dec:add(SBatchNorm(8*ngf)):add(nn.ReLU(true))
		-- state size : (8*ngf) x 4 x 4
		dec:add(SFullConv(8*ngf, 4*ngf, 4, 4, 2, 2, 1, 1):noBias())
		dec:add(SBatchNorm(4*ngf)):add(nn.ReLU(true))
		-- state size : (4*ngf) x 8 x 8
		dec:add(SFullConv(4*ngf, 2*ngf, 4, 4, 2, 2, 1, 1):noBias())
		dec:add(SBatchNorm(2*ngf)):add(nn.ReLU(true))
		-- state size : (2*ngf) x 16 x 16
		dec:add(SFullConv(2*ngf, ngf, 4, 4, 2, 2, 1, 1):noBias())
		dec:add(SBatchNorm(ngf)):add(nn.ReLU(true))
		-- state size : (ngf) x 32 x 32
		dec:add(SFullConv(ngf, ngf, 4, 4, 2, 2, 1, 1):noBias())
		dec:add(SBatchNorm(ngf)):add(nn.ReLU(true))
		-- state size : (ngf) x 64 x 64
		dec:add(SFullConv(ngf, nc, 4, 4, 2, 2, 1, 1))
		--dec:add(nn.Sigmoid())
		-- state size : (nc) x 128 x 128
        
        model:add(enc):add(dec)

	elseif type == 64 then
        -- encoder.
        -- state size : (3) x 64 x 64 
        local enc = nn.Sequential()
        enc:add(SConv(3, nef, 4, 4, 2, 2, 1, 1))
        enc:add(nn.ReLU(true))
        -- state size : (nef) x 32 x 32
        enc:add(SConv(nef, 2*nef, 4, 4, 2, 2, 1, 1))
        enc:add(SBatchNorm(2*nef)):add(nn.ReLU(true))
        -- state size : (2*nef) x 16 x 16
        enc:add(SConv(2*nef, 4*nef, 4, 4, 2, 2, 1, 1))
        enc:add(SBatchNorm(4*nef)):add(nn.ReLU(true))
        -- state size : (4*nef) x 8 x 8
        enc:add(SConv(4*nef, 8*nef, 4, 4, 2, 2, 1, 1))
        enc:add(SBatchNorm(8*nef)):add(nn.ReLU(true))
        -- state size : (8*nef) x 4 x 4
        enc:add(SConv(8*nef, nz, 4, 4))
        -- state size : (nz) x 1 x 1

        -- decoder.
		-- state size : (nBottleneck) x 1 x 1
		local dec = nn.Sequential()
        dec:add(SFullConv(nz, 8*ngf, 4, 4):noBias())
		dec:add(SBatchNorm(8*ngf)):add(nn.ReLU(true))
		-- state size : (8*ngf) x 4 x 4
		dec:add(SFullConv(8*ngf, 4*ngf, 4, 4, 2, 2, 1, 1):noBias())
		dec:add(SBatchNorm(4*ngf)):add(nn.ReLU(true))
		-- state size : (4*ngf) x 8 x 8
		dec:add(SFullConv(4*ngf, 2*ngf, 4, 4, 2, 2, 1, 1):noBias())
		dec:add(SBatchNorm(2*ngf)):add(nn.ReLU(true))
		-- state size : (2*ngf) x 16 x 16
		dec:add(SFullConv(2*ngf, ngf, 4, 4, 2, 2, 1, 1):noBias())
		dec:add(SBatchNorm(ngf)):add(nn.ReLU(true))
		-- state size : (ngf) x 32 x 32
		dec:add(SFullConv(ngf, nc, 4, 4, 2, 2, 1, 1))
		--dec:add(nn.Sigmoid())
		-- state size : (nc) x 64 x 64
        
        model:add(enc):add(dec)
	end

	return model
end


return Generator



