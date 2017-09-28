require 'nn'
require 'cunn'
local opts = require '__5_ebgan.script.opts'
local gen = require '__5_ebgan.models.gen'
local dis = require '__5_ebgan.models.dis'


-- basic settings.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(2)


-- get opt ans set seed.
local opt = opts.parse(arg)
print(opt)
if opt.seed == 0 then   opt.seed = torch.random(1,9999) end
torch.manualSeed(opt.seed)
print(string.format('Seed : %d', opt.seed))

-- set gpu.
if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.manualSeedAll(opt.seed)
    cutorch.setDevice(opt.gpuid+1)          -- lua index starts from 1 ...
end

-- create dataloader.
local loader = paths.dofile('../data/data.lua')


-- import trainer script.
require '__5_ebgan.script.ebgan'

-- load players(gen, dis)
local ebgan_models = {}
local ebgan_dis = dis.create_model(opt.sampleSize, opt)         -- discriminator
local ebgan_gen = gen.create_model(opt.sampleSize, opt)         -- generator    
ebgan_models = {ebgan_gen, ebgan_dis}
print ('Energy-Based GAN generator : ')    
print(ebgan_gen)
print ('Energy-Based GAN discriminator(enc/dec) : ')
print(ebgan_dis)

--loss metrics
local ebgan_criterion = {nn.AbsCriterion()}
--local ebgan_criterion = {nn.SmoothL1Criterion()}

-- run trainer
local optimstate = {}
local ebgan_trainer = EBGAN(ebgan_models, ebgan_criterion, opt, optimstate)
ebgan_trainer:train(10000, loader)


print('Congrats! You just finished the training.')













  
