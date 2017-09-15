
-- For training loop and learning rate scheduling.
-- This code is based on ResNet of Facebook, Inc.
-- last modified : 2017.07.21, nashory


require 'sys'
require 'optim'
require 'image'
require 'math'
local optimizer = require '__1_context-encoder.script.optimizer'


local trainer = torch.class('trainer')


function trainer:__init(model, criterion, opt, optimstate)
    self.model = model
    self.criterion = criterion
    self.optimstate = optimstate or {
        lr = opt.lr,
    }
    self.opt = opt
    self.noisetype = opt.noisetype
    self.nc = opt.nc
    self.nz = opt.nz
    self.batchSize = opt.batchSize
    self.sampleSize = opt.sampleSize

    if opt.display then
        self.disp = require 'display'
        self.disp.configure({hostname=opt.display_server_ip, port=8000})
    end

    -- get models and criterion.
    self.gen = model[1]:cuda()
    self.dis = model[2]:cuda()
    self.BCEcrit = criterion[1]:cuda()
    self.MSEcrit = criterion[2]:cuda()
end

function trainer:mask_generator(option)
    local mask
    if option == 'random' then
        print('random mask generation')
        local res = 0.06
        local MAX_SIZE = 1000
        local density = 0.25
        local low_mask = torch.Tensor(tonumber(res*MAX_SIZE), tonumber(res*MAX_SIZE)):uniform(0,1):mul(255)
        mask = image.scale(low_mask, MAX_SIZE, MAX_SIZE, 'bicubic')
        low_mask = nil
        mask:div(255)
        mask = torch.lt(mask, density):byte()       --25% 1s and 75% 0s
        mask = mask:byte()
    end

    return mask
end

trainer['fDx'] = function(self, x)
    self.dis:zeroGradParameters()
    
    -- train with real(x)
    self.x = self.dataset:getBatch()
    self.x_ctx = self.x:clone()
    -- get ranom mask
    local mask
    while true do
        local x = math.ceil(torch.uniform(1, self.MAX_SIZE-self.sampleSize))
        local y = math.ceil(torch.uniform(1, self.MAX_SIZE-self.sampleSize))
        mask = self.mask[{{y, y+self.sampleSize-1}, {x, x+self.sampleSize-1}}]:clone()
        local area = mask:sum()/(self.sampleSize*self.sampleSize)*100
        if area>20 and area<30 then         -- want it to be approx. 75% 0s and 25% 1s
            break
        end
    end
    self.curmask = mask:clone()
    torch.repeatTensor(self.mask_global, mask, self.batchSize, 1, 1)
    self.x_ctx[{{},1,{},{}}][self.mask_global] = 2*117.0/255.0 - 1.0
    self.x_ctx[{{},2,{},{}}][self.mask_global] = 2*104.0/255.0 - 1.0
    self.x_ctx[{{},3,{},{}}][self.mask_global] = 2*123.0/255.0 - 1.0
    -- forward and backward
    self.label:fill(1)      -- real label (1)
    local output_real = self.dis:forward(self.x:cuda())
    local errD_real = self.BCEcrit:forward(output_real:cuda(), self.label:cuda())
    local d_crit_real = self.BCEcrit:backward(output_real:cuda(), self.label:cuda())
    local d_dis_real = self.dis:backward(self.x:cuda(), d_crit_real:cuda())


    -- train with fake(x_tilde)
    self.x_tilde = self.gen:forward(self.x_ctx:cuda())
    self.label:fill(0)      -- fake label (0)
    
    -- forward and backward
    local output_fake = self.dis:forward(self.x_tilde:cuda())
    local errD_fake = self.BCEcrit:forward(output_fake:cuda(), self.label:cuda())
    local d_crit_fake = self.BCEcrit:backward(output_fake:cuda(), self.label:cuda())
    local d_dis_fake = self.dis:backward(self.x_tilde:cuda(), d_crit_fake:cuda())

    -- return error.
    local errD = errD_real + errD_fake
    return errD
end


trainer['fGx'] = function(self, x)
    self.gen:zeroGradParameters()

    self.label:fill(1)          -- fake labels are real for generator cost. (parameterization trick)

    local errG_adv = self.BCEcrit:forward(self.dis.output:cuda(), self.label:cuda())
    local d_crit_fake = self.BCEcrit:backward(self.dis.output:cuda(), self.label:cuda())
    local d_dis_fake = self.dis:backward(self.x_tilde:cuda(), d_crit_fake:cuda())
    self.d_gen_fake = self.gen:backward(self.x_ctx:cuda(), d_dis_fake:cuda())

    -- reconstruction loss
    local errG
    if (self.opt.recon_loss) then
        local recon_mask = torch.repeatTensor(self.mask_global:clone():mul(-1):add(1), 3, 1, 1, 1):permute(2,1,3,4)
        local fake_hole = self.x_tilde:clone()
        local real_hole = self.x:clone() 
        fake_hole[recon_mask] = 0
        real_hole[recon_mask] = 0

        local errG_recon = self.MSEcrit:forward(fake_hole:cuda(), real_hole:cuda())
        local d_recon_crit_fake = self.MSEcrit:forward(fake_hole:cuda(), real_hole:cuda())
        errG = 0.7*errG_adv + 0.3*errG_recon
    else
        errG = errG_adv
    end
    
    return errG
end


function trainer:train(epoch, loader)
    -- Initialize data variables.
    self.MAX_SIZE = 1000
    self.curmask = nil
    self.mask_global = torch.ByteTensor(self.batchSize, self.sampleSize, self.sampleSize):zero()
    self.label = torch.Tensor(self.batchSize):zero()
    self.noise = torch.Tensor(self.batchSize, self.nz, 1, 1)


    -- get network weights.
    self.dataset = loader.new(self.opt.nthreads, self.opt)
    print(string.format('Dataset size :  %d', self.dataset:size()))
    self.gen:training()
    self.dis:training()
    self.param_gen, self.gradParam_gen = self.gen:getParameters()
    self.param_dis, self.gradParam_dis = self.dis:getParameters()


    local totalIter = 0
    for e = 1, epoch do
        -- generate mask every epoch.
        self.mask = self:mask_generator('random')
        -- get max iteration for 1 epcoh.
        local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
        for iter  = 1, iter_per_epoch do
            totalIter = totalIter + 1
            -- forward/backward and update weights with optimizer.
            -- DO NOT CHANGE OPTIMIZATION ORDER.
            local err_dis, dx_dis = self:fDx()
            local err_gen, dx_gen = self:fGx()

            local gen_lr = 0
            if self.opt.recon_loss then gen_lr = 10*optimizer.gen.config.lr else gen_lr = optimizer.gen.config.lr end
            optimizer.dis.method(self.param_dis, self.gradParam_dis, optimizer.dis.config.lr,
                                optimizer.dis.config.beta1, optimizer.dis.config.beta2,
                                optimizer.dis.config.elipson, optimizer.dis.optimstate)
            optimizer.gen.method(self.param_gen, self.gradParam_gen, gen_lr,
                                optimizer.gen.config.beta1, optimizer.gen.config.beta2,
                                optimizer.gen.config.elipson, optimizer.gen.optimstate)

            -- save model at every sepficied epoch.
            local data = {dis = self.dis, gen = self.gen}
            self:snapshot(string.format('__1_context-encoder/repo/%s', self.opt.name), self.opt.name, totalIter, data)
            
            
            
            -- display server.
            if (totalIter%self.opt.display_iter==0) and (self.opt.display) then
                local x = self.x:clone():float()
                local x_tilde = self.x_tilde:clone():float()
                local x_ctx = self.x:clone():float()
                x_ctx[{{},1,{},{}}][self.mask_global] = x_tilde[{{},1,{},{}}][self.mask_global]
                x_ctx[{{},2,{},{}}][self.mask_global] = x_tilde[{{},2,{},{}}][self.mask_global]
                x_ctx[{{},3,{},{}}][self.mask_global] = x_tilde[{{},3,{},{}}][self.mask_global]

                self.disp.image(x, {win=self.opt.display_id + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(x_tilde, {win=self.opt.display_id*5 + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(x_ctx, {win=self.opt.display_id*10 + self.opt.gpuid, title=self.opt.server_name})
            end

            -- logging
            local log_msg = string.format('Epoch: [%d][%6d/%6d]\tLoss_D: %.4f\tLoss_G: %.4f', e, iter, iter_per_epoch, err_dis, err_gen)
            print(log_msg)
        end
    end
end

function trainer:snapshot(path, fname, iter, data)
    -- if dir not exist, create it.
    if not paths.dirp(path) then os.execute(string.format('mkdir -p %s', path)) end

    local fname = fname .. '_Iter' .. iter .. '.t7'
    local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
    if iter % math.ceil(self.opt.snapshot_every*iter_per_epoch) == 0 then
        local save_path = path .. '/' .. fname
        torch.save(save_path)
        print('[Snapshot]: saved model @ ' .. save_path)
    end
end


return trainer




