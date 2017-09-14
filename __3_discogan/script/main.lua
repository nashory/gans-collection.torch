require 'nn'
require 'cudnn'
local opts = require '__3_discogan.script.opts'
local gen = require '__3_discogan.models.gen'
local dis = require '__3_discogan.models.dis'


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
	cutorch.manualSeedAll(opt.seed)
	cutorch.setDevice(opt.gpuid+1)			-- lua index starts from 1 ...
end

-- create dataloader.
local loader = paths.dofile('../data/data.lua')


-- import trainer script.
require '__3_discogan.script.discogan'

-- load players(gen, dis)
local discogan_models = {}
local discogan_dis_doma = dis.create_model(opt.sampleSize, opt)			-- discriminator (domain a)
local discogan_gen_doma = gen.create_model(opt.sampleSize, opt)			-- generator (domain a)
local discogan_dis_domb = dis.create_model(opt.sampleSize, opt)			-- discriminator (domain b)
local discogan_gen_domb = gen.create_model(opt.sampleSize, opt)			-- generator (domain b)
discogan_models = {discogan_gen_doma, discogan_dis_doma, discogan_gen_domb, discogan_dis_domb}
print ('DiscoGAN generator : ')	
print(discogan_gen_doma)
print ('DiscoGAN discriminator : ')
print(discogan_dis_doma)

--loss metrics
local discogan_criterion = {nn.BCECriterion(), nn.MSECriterion()}

-- run trainer
local optimstate = {}
local discogan_trainer = DiscoGAN(discogan_models, discogan_criterion, opt, optimstate)
discogan_trainer:train(10000, loader)


print('Congrats! You just finished the training.')













  
