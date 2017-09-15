-- For training loop and learning rate scheduling.
-- standard DCGAN.
-- last modified : 2017.08.31, nashory


require 'sys'
require 'optim'
require 'image'
require 'math'
local optimizer = require '__0_dcgan.script.optimizer'


local DCGAN = torch.class('DCGAN')


function DCGAN:__init(model, criterion, opt, optimstate)
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
    
    -- generate test_noise(fixed)
    self.test_noise = torch.Tensor(self.batchSize, self.nz, 1, 1)
    if self.noisetype == 'uniform' then self.test_noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.test_noise:normal(0,1) end
    
    if opt.display then
        self.disp = require 'display'
        self.disp.configure({hostname=opt.display_server_ip, port=opt.display_server_port})
    end

    -- get models and criterion.
    self.gen = model[1]:cuda()
    self.dis = model[2]:cuda()
    self.BCEcrit = criterion[1]:cuda()
    self.MSEcrit = criterion[2]:cuda()
end

DCGAN['fDx'] = function(self, x)
    self.dis:zeroGradParameters()
    
    -- generate noise(z)
    if self.noisetype == 'uniform' then self.noise:uniform(-1,1)
    elseif self.noisetype == 'normal' then self.noise:normal(0,1) end
    
    -- train with real(x)
    self.real_data = self.dataset:getBatch()
    self.x = self.real_data:clone()
    -- forward and backward
    self.label:fill(1)      -- real label (1)
    local f_x = self.dis:forward(self.x:cuda()):clone()
    local errD_real = self.BCEcrit:forward(f_x:cuda(), self.label:cuda())
    local d_real_crit = self.BCEcrit:backward(f_x:cuda(), self.label:cuda())
    local d_real_dis = self.dis:backward(self.x:cuda(), d_real_crit:cuda())

    -- train with fake(x_tilde)
    self.z = self.noise:clone()
    self.label:fill(0)      -- fake label (0)
    -- forward and backward
    self.x_tilde = self.gen:forward(self.z:cuda()):clone()
    self.f_x_tilde = self.dis:forward(self.x_tilde:cuda()):clone()
    local errD_fake = self.BCEcrit:forward(self.f_x_tilde:cuda(), self.label:cuda())
    local d_fake_crit = self.BCEcrit:backward(self.f_x_tilde:cuda(), self.label:cuda())
    local d_fake_dis = self.dis:backward(self.x_tilde:cuda(), d_fake_crit:cuda())

    -- return error.
    local errD = errD_real + errD_fake
    return errD
end


DCGAN['fGx'] = function(self, x)
    self.gen:zeroGradParameters()

    self.label:fill(1)          -- fake labels are real for generator cost. (since we prefer maximizing D(G(x)), instead of minimizing 1-D(G(x)).)
    local errG = self.BCEcrit:forward(self.f_x_tilde:cuda(), self.label:cuda())
    local d_gen_crit = self.BCEcrit:backward(self.f_x_tilde:cuda(), self.label:cuda())
    local d_gen_dis = self.dis:updateGradInput(self.x_tilde:cuda(), d_gen_crit:cuda())
    local d_gen_dummy = self.gen:backward(self.z:cuda(), d_gen_dis:cuda())

    return errG
end


function DCGAN:train(epoch, loader)
    -- Initialize data variables.
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
        -- get max iteration for 1 epcoh.
        local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
        for iter  = 1, iter_per_epoch do
            totalIter = totalIter + 1

            -- forward/backward and update weights with optimizer.
            -- DO NOT CHANGE OPTIMIZATION ORDER.
            local err_dis = self:fDx()
            local err_gen = self:fGx()

            -- weight update.
            optimizer.dis.method(self.param_dis, self.gradParam_dis, optimizer.dis.config.lr,
                                optimizer.dis.config.beta1, optimizer.dis.config.beta2,
                                optimizer.dis.config.elipson, optimizer.dis.optimstate)
            optimizer.gen.method(self.param_gen, self.gradParam_gen, optimizer.gen.config.lr,
                                optimizer.gen.config.beta1, optimizer.gen.config.beta2,
                                optimizer.gen.config.elipson, optimizer.gen.optimstate)

            -- save model at every specified epoch.
            local data = {dis = self.dis, gen = self.gen}
            self:snapshot(string.format('__0_dcgan/repo/%s', self.opt.name), self.opt.name, totalIter, data)

            -- display server.
            if (totalIter%self.opt.display_iter==0) and (self.opt.display) then
                local im_fake = self.x_tilde:clone()
                local im_real = self.x:clone()
                
                self.disp.image(im_real, {win=self.opt.display_id + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(im_fake, {win=self.opt.display_id*5 + self.opt.gpuid, title=self.opt.server_name})


                -- save image as png (size 64x64, grid 8x8 fixed).
                local im_png = torch.Tensor(3, self.sampleSize*8, self.sampleSize*8):zero()
                local x_test = self.gen:forward(self.test_noise:cuda())
                for i = 1, 8 do
                    for j =  1, 8 do
                        im_png[{{},{self.sampleSize*(j-1)+1, self.sampleSize*(j)},{self.sampleSize*(i-1)+1, self.sampleSize*(i)}}]:copy(x_test[{{8*(i-1)+j},{},{},{}}]:clone():add(1):div(2))
                    end
                end
                os.execute('mkdir -p __0_dcgan/repo/image')
                image.save(string.format('__0_dcgan/repo/image/%d.jpg', totalIter/self.opt.display_iter), im_png)
            end

            -- logging
            local log_msg = string.format('Epoch: [%d][%6d/%6d]\tLoss_D: %.4f\tLoss_G: %.4f', e, iter, iter_per_epoch, err_dis, err_gen)
            print(log_msg)
        end
    end
end


function DCGAN:snapshot(path, fname, iter, data)
    -- if dir not exist, create it.
    if not paths.dirp(path) then    os.execute(string.format('mkdir -p %s', path)) end
    
    local fname = fname .. '_Iter' .. iter .. '.t7'
    local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
    if iter % math.ceil(self.opt.snapshot_every*iter_per_epoch) == 0 then
        local save_path = path .. '/' .. fname
        torch.save(save_path)
        print('[Snapshot]: saved model @ ' .. save_path)
    end
end


return DCGAN




