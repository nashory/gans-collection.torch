-- optimizer

require 'torch'
require 'script.optim_updates'
local opts = require 'script.opts'


local optimizer = {}
local opt = opts.parse(arg)

-- discriminator optimizer.
optimizer.dis = {
	method = adam,
	config = {
		lr = opt.lr,
		beta1 = 0.5,
		beta2 = 0.997,
		elipson = 1e-8,
	},
	optimstate = {}
}

-- generator optimizer
optimizer.gen = {
	method = adam,
	config = {
		lr = opt.lr,
		beta1 = 0.5,
		beta2 = 0.997,
		elipson = 1e-8,
	},
	optimstate = {}
}

return optimizer










