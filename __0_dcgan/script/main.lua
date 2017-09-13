require 'nn'
local opts = require '__0_dcgan.script.opts'
local gen = require '__0_dcgan.models.gen'
local dis = require '__0_dcgan.models.dis'


-- basic settings.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(2)


-- get opt ans set seed.
local opt = opts.parse(arg)
print(opt)
if opt.seed == 0 then	opt.seed = torch.random(1,9999)	end
torch.manualSeed(opt.seed)
print(string.format('Seed : %d', opt.seed))

-- set gpu.
if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	require 'cudnn'
	cutorch.manualSeedAll(opt.seed)
	cutorch.setDevice(opt.gpuid+1)			-- lua index starts from 1 ...
end

-- create dataloader.
local loader = paths.dofile('../../data/data.lua')


-- import trainer script.
require '__0_dcgan.script.dcgan'

-- load players(gen, dis)
local dcgan_models = {}
local dcgan_dis = dis.create_model(opt.sampleSize, 'DCGAN', opt)			-- discriminator
local dcgan_gen = gen.create_model(opt.sampleSize, opt)						-- generator	
dcgan_models = {dcgan_gen, dcgan_dis}
print ('DCGAN generator : ')	
print(dcgan_gen)
print ('DCGAN discriminator : ')
print(dcgan_dis)

--loss metrics
local dcgan_criterion = {nn.BCECriterion(), nn.MSECriterion()}

-- run trainer
local optimstate = {}
local dcgan_trainer = DCGAN(dcgan_models, dcgan_criterion, opt, optimstate)
dcgan_trainer:train(10000, loader)


print('Congrats! You just finished the training.')













  
