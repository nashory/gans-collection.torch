require 'nn'
local opts = require '__1_context-encoder.script.opts'


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
require '__1_context-encoder.script.context-encoder'
local gen = require '__1_context-encoder.models.gen'
local dis = require '__1_context-encoder.models.dis'

-- load players(gen, dis).
local gan_models = {}
local gan_gen = gen.create_model(opt.sampleSize, opt)		-- fake image generator.
local gan_dis = dis.create_model(opt.sampleSize, opt)		-- discriminator.
gan_models = {gan_gen, gan_dis}
print('context-encoder generator : ')
print(gan_gen)
print('context-encoder discriminator : ')
print(gan_dis)

-- loss metrics
local gan_criterion = {nn.BCECriterion(), nn.MSECriterion()}

-- run trainer
local gan_trainer = trainer(gan_models, gan_criterion, opt)
gan_trainer:train(10000, loader)

print('Congrats! You just finished the training.')













  
