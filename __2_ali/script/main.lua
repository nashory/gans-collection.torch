require 'nn'
local opts = require '__2_ali.script.opts'


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
require '__2_ali.script.ali'
local gen = require '__2_ali.models.gen'
local enc = require '__2_ali.models.enc'
local dis = require '__2_ali.models.dis'

-- load players(enc, gen, dis).
local ali_models = {}
local ali_enc = enc.create_model(opt.sampleSize, opt)            -- latent variable encoder.
local ali_gen = gen.create_model(opt.sampleSize, opt)            -- fake image generator.
local ali_dis = dis.create_model(opt.sampleSize, opt)            -- joint(x,z) discriminator.
ali_models = {ali_enc, ali_gen, ali_dis}
print('ALI encoder : ')
print(ali_enc)
print('ALI generator : ')
print(ali_gen)
print('ALI discriminator : ')
print(ali_dis)

-- loss metrics
local ali_criterion = {nn.BCECriterion(), nn.MSECriterion()}

-- run trainer
local optimstate = {}
local ali_trainer = ALI(ali_models, ali_criterion, opt, optimstate)
ali_trainer:train(10000, loader)

print('Congrats! You just finished the training.')













  
