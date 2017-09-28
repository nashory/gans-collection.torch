-- Generator network structure for BEGAN.


require 'nn'
require 'cudnn'
require 'math'

local Generator = {}

local SBatchNorm = cudnn.SpatialBatchNormalization
local SConv = cudnn.SpatialConvolution
local SFullConv = cudnn.SpatialFullConvolution
local ELU = nn.ELU
local UpSampleNearest = nn.SpatialUpSamplingNearest
local Linear = nn.Linear


-- Generator input context to noise
function Generator.create_model(type, opt)
	assert(type%16==0, 'erorr. type argument must be multiples of 2 and larger than 2^3.')

	local nc = opt.nc
	local nh = opt.nh
	local ngf = opt.ngf
	local model = nn.Sequential() 
    
    -- state size : (h)
    model:add(Linear(nh, 2048))
    model:add(nn.Reshape(32, 8, 8))
    -- state size : (n x 8 x 8)

    local rep = math.log(type, 2) - 3
    for i = 1, rep do
        local ns = (i-1)*ngf
        if i == 1 then ns = 32 end
        model:add(SConv(ns, i*ngf, 3, 3, 1, 1, 1, 1))
        model:add(ELU())
        --model:add(SBatchNorm(i*ngf)):add(ELU())
        model:add(SConv(i*ngf, i*ngf, 3, 3, 1, 1, 1, 1))
        --model:add(SBatchNorm(i*ngf)):add(ELU())
        model:add(ELU())
        model:add(UpSampleNearest(2.0))
    end
    model:add(SConv(rep*ngf, nc, 3, 3, 1, 1, 1, 1))
    model:add(nn.Tanh())

	return model
end


return Generator





