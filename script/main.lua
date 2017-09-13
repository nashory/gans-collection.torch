require 'nn'
local opts = require 'script.opts'
local gen = require 'models.gen'
local enc = require 'models.enc'
local dis = require 'models.dis'


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

if opt.GANtype == 'ALI' then
	-- import trainer script.
	require 'script.ALI'

	-- load players(enc, gen, dis).
	local ali_models = {}
	local ali_enc = enc.create_model(opt.sampleSize, opt)					-- latent variable encoder.
	local ali_gen = gen.create_model(opt.sampleSize, opt)					-- fake image generator.
	local ali_dis = dis.create_model(opt.sampleSize, 'ALI', opt)			-- joint(x,z) discriminator.
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

elseif opt.GANtype == 'DCGAN' then
	-- import trainer script.
	require 'script.DCGAN'

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
end



print('Congrats! You just finished the training.')













  
