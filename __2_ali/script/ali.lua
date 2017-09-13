-- For training loop and learning rate scheduling.
-- ALI (Adversarial Learning Inference)
-- last modified : 2017.09.01, nashory


require 'sys'
require 'optim'
require 'image'
require 'math'
local optimizer = require '__2_ali.script.optimizer'


local ALI = torch.class('ALI')


function ALI:__init(model, criterion, opt, optimstate)
	self.model = model
	self.criterion = criterion
	self.optimstate = optimstate or {
		lr = opt.lr,
		momentum = opt.momentum,
		weight_decay = opt.weight_decay,
	}
	self.opt = opt
	self.noisetype = opt.noisetype
	self.nc = opt.nc
	self.nz = opt.nz
	self.batchSize = opt.batchSize
	self.sampleSize = opt.sampleSize

	if opt.display then
		self.disp = require 'display'
		self.disp.configure({hostname=opt.display_server_ip, port=opt.display_server_port})
	end

	-- get models and criterion.
	self.enc = model[1]:cuda()
	self.gen = model[2]:cuda()
	self.dis = model[3]:cuda()
	self.BCEcrit = criterion[1]:cuda()
	self.MSEcrit = criterion[2]:cuda()
end


ALI['fDx'] = function(self)
	self.dis:zeroGradParameters()
	
	-- generate noise(z)
	if self.noisetype == 'uniform' then self.noise:uniform(-1,1)
	elseif self.noisetype == 'normal' then self.noise:normal(0,1) end

	-- concat input (backward real and fake simultaneously.)
	-- real data (x, z_hat)
	self.real_data = self.dataset:getBatch()
	self.x = self.real_data:clone()
	self.z_hat = self.enc:forward(self.x:cuda()):clone()
	-- fake data (x_tilde, z)
	self.z = self.noise:clone()
	self.x_tilde = self.gen:forward(self.z:cuda()):clone()
	-- prepare input ({x, x_tilde}, {z_hat, z}).
	self.input_xs = torch.cat(self.x:cuda(), self.x_tilde:cuda(), 1)
	self.input_zs = torch.cat(self.z_hat:cuda(), self.z:cuda(), 1)
	self.label[{{1, self.batchSize}}]:fill(1)						-- real label(1) for x, z_hat
	self.label[{{self.batchSize+1, 2*self.batchSize}}] = 0			-- fake label(0) for x_tilde, z
	
	--forward and backward
	self.f_dis = self.dis:forward({self.input_xs:cuda(), self.input_zs:cuda()}):clone()
	local errD = self.BCEcrit:forward(self.f_dis:cuda(), self.label:cuda())
	local d_crit = self.BCEcrit:backward(self.f_dis:cuda(), self.label:cuda())
	local d_dis = self.dis:backward({self.input_xs:cuda(), self.input_zs:cuda()}, d_crit)

	return errD
end


ALI['fGx'] = function(self)
	self.gen:zeroGradParameters()
	self.label[{{1, self.batchSize}}] = 0						-- fake label(0) for x, z_hat
	self.label[{{self.batchSize+1, 2*self.batchSize}}] = 1		-- real label(1) for x_tilde, z
	-- forward and backward
	local errG = self.BCEcrit:forward(	self.f_dis[{{self.batchSize+1, 2*self.batchSize}}]:cuda(), 
										self.label[{{self.batchSize+1, 2*self.batchSize}}]:cuda())
	local d_crit = self.BCEcrit:updateGradInput(self.f_dis:cuda(), self.label:cuda())
	local d_dis = self.dis:updateGradInput({self.input_xs:cuda(), self.input_zs:cuda()}, d_crit:cuda())
	local d_gen = self.gen:backward(self.z:cuda(), d_dis[1][{{1+self.batchSize, 2*self.batchSize},{},{},{}}]:cuda())
	return errG
end


ALI['fEx'] = function(self)
	self.enc:zeroGradParameters()	
	self.label[{{1, self.batchSize}}] = 0						-- fake label(0) for x, z_hat
	self.label[{{self.batchSize+1, 2*self.batchSize}}] = 1		-- real label(1) for x_tilde, z
	-- forward and backward
	local errE = self.BCEcrit:forward(	self.f_dis[{{1, self.batchSize}}]:cuda(), 
										self.label[{{1, self.batchSize}}]:cuda())
	local d_crit = self.BCEcrit:updateGradInput(self.f_dis:cuda(), self.label:cuda())
	local d_dis = self.dis:updateGradInput({self.input_xs:cuda(), self.input_zs:cuda()}, d_crit:cuda())
	local d_enc = self.enc:backward(self.x:cuda(), d_dis[2][{{1, self.batchSize},{},{},{}}]:cuda())
	return errE

end


function ALI:train(epoch, loader)
	-- Initialize data variables.
	self.label = torch.Tensor(2*self.batchSize):zero()
	self.noise = torch.Tensor(self.batchSize, self.nz, 1, 1)

	-- get network weights.
	self.dataset = loader.new(self.opt.nthreads, self.opt)
	print(string.format('Dataset size :  %d', self.dataset:size()))
	self.enc:training()
	self.gen:training()
	self.dis:training()
	self.param_enc, self.gradParam_enc = self.enc:getParameters()
	self.param_gen, self.gradParam_gen = self.gen:getParameters()
	self.param_dis, self.gradParam_dis = self.dis:getParameters()


	local totalIter = 0
	for e = 1, epoch do
		-- get max iteration for 1 epcoh.
		local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
		for iter  = 1, iter_per_epoch do
			totalIter = totalIter + 1

			-- forward/backward and update weights with optimizer.
			-- DO NOT CHANGE OPTIMIZATION ORDER.
			local err_dis = self:fDx()
			local err_enc = self:fEx()
			local err_gen = self:fGx()

			-- weight update.
			optimizer.dis.method(self.param_dis, self.gradParam_dis, optimizer.dis.config.lr,
								optimizer.dis.config.beta1, optimizer.dis.config.beta2,
								optimizer.dis.config.elipson, optimizer.dis.optimstate)
			optimizer.gen.method(self.param_gen, self.gradParam_gen, optimizer.gen.config.lr,
								optimizer.gen.config.beta1, optimizer.gen.config.beta2,
								optimizer.gen.config.elipson, optimizer.gen.optimstate)
			optimizer.enc.method(self.param_enc, self.gradParam_enc, optimizer.enc.config.lr,
								optimizer.enc.config.beta1, optimizer.enc.config.beta2,
								optimizer.enc.config.elipson, optimizer.enc.optimstate)


			-- save model at every specified epoch.
			local data = {dis = self.dis, gen = self.gen, enc = self.enc}
			self:snapshot(string.format('__2_ali/repo/%s', self.opt.name), self.opt.name, totalIter, data)

			-- display server.
			if (totalIter%self.opt.display_iter==0) and (self.opt.display) then
				local real_data = self.dataset:getBatch()	
				local z_hat = self.enc:forward(real_data:cuda())
				local z = self.noise:clone()
				local im_fake_from_z_hat = self.gen:forward(z_hat:cuda()):clone()
				local im_real = real_data:clone()
				local im_fake_from_z = self.gen:forward(z:cuda()):clone()
				
				self.disp.image(im_real, {win=self.opt.display_id + self.opt.gpuid, title=self.opt.server_name})
				self.disp.image(im_fake_from_z_hat, {win=self.opt.display_id*5 + self.opt.gpuid, title=self.opt.server_name})
				self.disp.image(im_fake_from_z, {win=self.opt.display_id*10 + self.opt.gpuid, title=self.opt.server_name})
			end

			-- logging
			local log_msg = string.format('Epoch: [%d][%6d/%6d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_E: %.4f', e, iter, iter_per_epoch, err_dis, err_gen, err_enc)
			print(log_msg)
		end
	end
end


function ALI:snapshot(path, fname, iter, data)
	-- if dir not exist, create it.
	if not paths.dirp(path) then	os.execute(string.format('mkdir -p %s', path)) end
	
	local fname = fname .. '_Iter' .. iter .. '.t7'
	local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
	if iter % math.ceil(self.opt.snapshot_every*iter_per_epoch) == 0 then
		local save_path = path .. '/' .. fname
		torch.save(save_path)
		print('[Snapshot]: saved model @ ' .. save_path)
	end
end

return ALI




