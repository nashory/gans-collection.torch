require 'nn'
require 'cudnn'
local opts = require '__4_cyclegan.script.opts'
local gen = require '__4_cyclegan.models.gen'
local dis = require '__4_cyclegan.models.dis'


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
require '__4_cyclegan.script.cyclegan'

-- load players(gen, dis)
local cyclegan_models = {}
local cyclegan_dis_doma = dis.create_model(opt.sampleSize, opt)			-- discriminator (domain a)
local cyclegan_gen_doma = gen.create_model(opt.sampleSize, opt)			-- generator (domain a)
local cyclegan_dis_domb = dis.create_model(opt.sampleSize, opt)			-- discriminator (domain b)
local cyclegan_gen_domb = gen.create_model(opt.sampleSize, opt)			-- generator (domain b)
cyclegan_models = {cyclegan_gen_doma, cyclegan_dis_doma, cyclegan_gen_domb, cyclegan_dis_domb}
print ('CycleGAN generator : ')	
print(cyclegan_gen_doma)
print ('CycleGAN discriminator : ')
print(cyclegan_dis_doma)


-- loss metrics:
-- used AbsCriterion for cycle-consistency loss to avoid blurry output.
-- paper states leaset-square(MSECriterion) performs better for adversarial loss.
local cyclegan_criterion = {}
if opt.use_lsgan then cyclegan_criterion = {nn.MSECriterion(), nn.AbsCriterion()}
else cyclegan_criterion = {nn.BCECriterion(), nn.AbsCriterion()} end


-- run trainer
local optimstate = {}
local cyclegan_trainer = CycleGAN(cyclegan_models, cyclegan_criterion, opt, optimstate)
cyclegan_trainer:train(10000, loader)


print('Congrats! You just finished the training.')













  
