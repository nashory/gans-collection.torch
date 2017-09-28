-- Discriminator network structure for BEGAN.


local nn = require 'nn'
local cudnn = require 'cudnn'


local Discrim = {}

local SBatchNorm = cudnn.SpatialBatchNormalization
local SConv = cudnn.SpatialConvolution
local SFullConv = cudnn.SpatialFullConvolution
local LeakyReLU = nn.LeakyReLU
local ELU = nn.ELU
local UpSampleNearest = nn.SpatialUpSamplingNearest
local UpSampleBilinear = nn.SpatialUpSamplingBilinear
local Linear = nn.Linear
local AvgPool = nn.SpatialAveragePooling


-- Encode input context to noise
function Discrim.create_model(type, opt)
	assert(	type%16==0, 'error. type argument must be multiples of 2 and larger than 2^3.') 

	local nc = opt.nc
    local nh = opt.nh
	local nz = opt.nz
    local batchSize = opt.batchSize
	local ndf = opt.ndf
	local model = nn.Sequential()
    local enc = nn.Sequential()
    local dec = nn.Sequential()
    local rep = math.log(type, 2) - 3

    -- Encoder.
    -- state size : (nc) x type x type
    enc:add(SConv(nc, ndf, 3, 3, 1, 1, 1, 1))
    enc:add(ELU())
    -- state size : (ndf) x type x type
    for i=1, rep do
        enc:add(SConv(i*ndf, i*ndf, 3, 3, 1, 1, 1, 1))
        enc:add(ELU())
        --enc:add(SBatchNorm(i*ndf)):add(ELU())
        enc:add(SConv(i*ndf, (i+1)*ndf, 3, 3, 1, 1, 1, 1))
        --enc:add(SBatchNorm((i+1)*ndf)):add(ELU())
        enc:add(ELU())
        enc:add(AvgPool(2, 2, 2, 2))            -- downsampling by factor of 2.
        --[[
        if i == rep then
            enc:add(SConv(i*ndf, (i+1)*ndf, 3, 3, 2, 2, 1, 1))
        else 
            enc:add(SConv(i*ndf, (i+1)*ndf, 3, 3, 2, 2, 1, 1))    -- use Conv for downsampling instead of MaxPool
            enc:add(ELU())
        end
        ]]--
    end
    -- state size : (ndf*rep) x 8 x 8
    enc:add(SConv((rep+1)*ndf, (rep+1)*ndf, 3, 3, 1, 1, 1, 1))
    --enc:add(SBatchNorm((rep+1)*ndf)):add(ELU())
    enc:add(ELU())
    enc:add(SConv((rep+1)*ndf, 32, 3, 3, 1, 1, 1, 1))       -- we fix output conv units to 64 to reduce memory usage. (64x8x8=4096 units)
    --enc:add(SBatchNorm(64)):add(ELU())
    enc:add(ELU())
    -- state size : (ndf*(rep+1)) x 8 x 8
    enc:add(nn.Reshape(2048))
    enc:add(Linear(2048, nh))
    -- state size : (nh)

    --Decoder.
    -- state size : (nh)
    dec:add(Linear(nh, 2048))
    dec:add(nn.Reshape(32, 8, 8))
    -- state size : (64 x 8 x 8)
    for i=1, rep do
        local ns = (i-1)*ndf
        if i==1 then ns = 32 end
        dec:add(SConv(ns, i*ndf, 3, 3, 1, 1, 1, 1))
        --dec:add(SBatchNorm((i)*ndf)):add(ELU())
        dec:add(ELU())
        dec:add(SConv(i*ndf, i*ndf, 3, 3, 1, 1, 1, 1))
        --dec:add(SBatchNorm((i)*ndf)):add(ELU())
        dec:add(ELU())
        dec:add(UpSampleNearest(2.0))
        --dec:add(UpSampleBilinear(2.0))
        --dec:add(SFullConv(i*ndf, i*ndf, 4, 4, 2, 2, 1, 1))
    end
    dec:add(SConv(rep*ndf, nc, 3, 3, 1, 1, 1, 1))
    --dec:add(nn.Tanh())

    -- combine model(enc, dec) and return.
    model:add(enc):add(dec)
    return model
end


return Discrim



