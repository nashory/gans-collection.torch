require 'nn'
local opts = require '__6_lsgan.script.opts'
local gen = require '__6_lsgan.models.gen'
local dis = require '__6_lsgan.models.dis'


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
local loader = paths.dofile('../data/data.lua')


-- import trainer script.
require '__6_lsgan.script.lsgan'

-- load players(gen, dis)
local lsgan_models = {}
local lsgan_dis = dis.create_model(opt.sampleSize, opt)			-- discriminator
local lsgan_gen = gen.create_model(opt.sampleSize, opt)		    -- generator	
lsgan_models = {lsgan_gen, lsgan_dis}
print ('LSGAN generator : ')	
print(lsgan_gen)
print ('LSGAN discriminator : ')
print(lsgan_dis)

--loss metrics
local lsgan_criterion = {nn.MSECriterion()}
--local lsgan_criterion = {nn.BCECriterion()}

-- run trainer
local optimstate = {}
local lsgan_trainer = LSGAN(lsgan_models, lsgan_criterion, opt, optimstate)
lsgan_trainer:train(10000, loader)


print('Congrats! You just finished the training.')













  
