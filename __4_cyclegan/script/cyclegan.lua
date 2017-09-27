-- For training loop and learning rate scheduling.
-- CycleGAN.
-- last modified : 2017.09.14, nashory


require 'sys'
require 'optim'
require 'image'
require 'math'
local optimizer = require '__4_cyclegan.script.optimizer'


local CycleGAN = torch.class('CycleGAN')


function CycleGAN:__init(model, criterion, opt, optimstate)
    self.model = model
    self.criterion = criterion
    self.optimstate = optimstate or {
        lr = opt.lr,
    }
    self.opt = opt
    self.nc = opt.nc
    self.nz = opt.nz
    self.batchSize = opt.batchSize
    self.sampleSize = opt.sampleSize

    if opt.display then
        self.disp = require 'display'
        self.disp.configure({hostname=opt.display_server_ip, port=opt.display_server_port})
    end

    -- get models and criterion.
    self.gen_domab = model[1]:cuda()
    self.dis_doma = model[2]:cuda()
    self.gen_domba = model[3]:cuda()
    self.dis_domb = model[4]:cuda()
    self.MSEcrit = criterion[1]:cuda()
    self.ABScrit = criterion[2]:cuda()
end


CycleGAN['fDx_doma'] = function(self)
    self.dis_doma:zeroGradParameters()
    self.x_doma = self.dataset:getBatchByClass(1)
    self.x_domb = self.dataset:getBatchByClass(2)
    
    -- train with real(x_doma)
    self.label:fill(1)-- real label (1)
    local f_x = self.dis_doma:forward(self.x_doma:cuda()):clone()
    local errD_real = self.MSEcrit:forward(f_x:cuda(), self.label:cuda())
    local d_real_crit = self.MSEcrit:backward(f_x:cuda(), self.label:cuda())
    local d_real_dis = self.dis_doma:backward(self.x_doma:cuda(), d_real_crit:cuda())

    -- train with fake (x_doma_tilde)
    self.label:fill(0)-- fake label (0)
    self.x_doma_tilde = self.gen_domba:forward(self.x_domb:cuda()):clone()
    self.f_x_doma_tilde = self.dis_doma:forward(self.x_doma_tilde:cuda()):clone()
    local errD_fake = self.MSEcrit:forward(self.f_x_doma_tilde:cuda(), self.label:cuda())
    local d_fake_crit = self.MSEcrit:backward(self.f_x_doma_tilde:cuda(), self.label:cuda())
    local d_fake_dis = self.dis_doma:backward(self.x_doma_tilde:cuda(), d_fake_crit:cuda())

    -- return error.
    local errD = errD_real + errD_fake
    return errD
end

CycleGAN['fDx_domb'] = function(self)
    self.dis_domb:zeroGradParameters()

    -- train with real(x_domb)
    self.label:fill(1)      -- real label (1)
    local f_x = self.dis_domb:forward(self.x_domb:cuda()):clone()
    local errD_real = self.MSEcrit:forward(f_x:cuda(), self.label:cuda())
    local d_real_crit = self.MSEcrit:backward(f_x:cuda(), self.label:cuda())
    local d_real_dis = self.dis_domb:backward(self.x_domb:cuda(), d_real_crit:cuda())

    -- train with fake(x_doma_tilde)
    self.label:fill(0)      -- fake label (0)
    self.x_domb_tilde = self.gen_domab:forward(self.x_doma:cuda()):clone()
    self.f_x_domb_tilde = self.dis_domb:forward(self.x_domb_tilde:cuda()):clone()
    local errD_fake = self.MSEcrit:forward(self.f_x_domb_tilde:cuda(), self.label:cuda())
    local d_fake_crit = self.MSEcrit:backward(self.f_x_domb_tilde:cuda(), self.label:cuda())
    local d_fake_dis = self.dis_domb:backward(self.x_domb_tilde:cuda(), d_fake_crit:cuda())

    -- return error.
    local errD = errD_real + errD_fake
    return errD
end


CycleGAN['fGx'] = function(self)
    self.gen_domab:zeroGradParameters()
    self.gen_domba:zeroGradParameters()

    -- train with adversrial (L_adv)
    -- gen_domab (G)
    self.label:fill(1)          -- fake labels are real for generator cost.
    local errG_adv = self.MSEcrit:forward(self.f_x_domb_tilde:cuda(), self.label:cuda())
    local d_gen_crit = self.MSEcrit:backward(self.f_x_domb_tilde:cuda(), self.label:cuda())
    local d_gen_dis = self.dis_domb:updateGradInput(self.x_domb_tilde:cuda(), d_gen_crit:cuda())
    local d_gen_dummy = self.gen_domab:backward(self.x_doma:cuda(), d_gen_dis:cuda())
    -- gen_domba (F)
    local errG_adv = self.MSEcrit:forward(self.f_x_doma_tilde:cuda(), self.label:cuda())
    local d_gen_crit = self.MSEcrit:backward(self.f_x_doma_tilde:cuda(), self.label:cuda())
    local d_gen_dis = self.dis_doma:updateGradInput(self.x_doma_tilde:cuda(), d_gen_crit:cuda())
    local d_gen_dummy = self.gen_domba:backward(self.x_domb:cuda(), d_gen_dis:cuda())
       
    -- train with reconstruction (L_cycle)
    local x_domb_cycle = self.gen_domab(self.x_doma_tilde:cuda())
    local x_doma_cycle = self.gen_domba(self.x_domb_tilde:cuda())
    local errG_cycle_domb = self.ABScrit:forward(x_domb_cycle:cuda(), self.x_domb:cuda())
    local d_cycle_domb = self.ABScrit:backward(x_domb_cycle:cuda(), self.x_domb:cuda())
    local errG_cycle_doma = self.ABScrit:forward(x_doma_cycle:cuda(), self.x_doma:cuda())
    local d_cycle_doma = self.ABScrit:backward(x_doma_cycle:cuda(), self.x_doma:cuda())
    local d_cycle_sum = torch.add(d_cycle_doma, d_cycle_domb)      -- eq (2) in section 3.2
    local d_gen_domab = self.gen_domab:backward(self.x_doma_tilde:cuda(), d_cycle_sum:mul(-1))
    local d_gen_dumba = self.gen_domba:backward(self.x_domb_tilde:cuda(), d_cycle_sum:mul(-1))
    local errG_cycle = (errG_cycle_domb + errG_cycle_doma)/2.0          -- avg loss.
    
    
    return errG_adv, errG_cycle
end


function CycleGAN:train(epoch, loader)
    -- Initialize data variables.
    self.label = torch.Tensor(self.batchSize, 1):zero()

    -- get network weights.
    self.dataset = loader.new(self.opt.nthreads, self.opt)
    print(string.format('Dataset size :  %d', self.dataset:size()))
    self.gen_domba:training()
    self.dis_doma:training()
    self.gen_domab:training()
    self.dis_domb:training()
    self.param_gen_domba, self.gradParam_gen_domba = self.gen_domba:getParameters()
    self.param_dis_doma, self.gradParam_dis_doma = self.dis_doma:getParameters()
    self.param_gen_domab, self.gradParam_gen_domab = self.gen_domab:getParameters()
    self.param_dis_domb, self.gradParam_dis_domb = self.dis_domb:getParameters()


    local totalIter = 0
    for e = 1, epoch do
        -- get max iteration for 1 epcoh.
        local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
        for iter  = 1, iter_per_epoch do
            totalIter = totalIter + 1

            -- forward/backward and update weights with optimizer.
            -- DO NOT CHANGE OPTIMIZATION ORDER.
            local err_dis_doma = self:fDx_doma()
            local err_dis_domb = self:fDx_domb()
            local err_gen_adv, err_gen_cycle = self:fGx()
            collectgarbage()

            -- weight update.
            optimizer.dis.method(self.param_dis_doma, self.gradParam_dis_doma, optimizer.dis.config.lr,
                                optimizer.dis.config.beta1, optimizer.dis.config.beta2,
                                optimizer.dis.config.elipson, optimizer.dis.optimstate.doma)
            optimizer.gen.method(self.param_gen_domba, self.gradParam_gen_domba, optimizer.gen.config.lr,
                                optimizer.gen.config.beta1, optimizer.gen.config.beta2,
                                optimizer.gen.config.elipson, optimizer.gen.optimstate.domba)
            optimizer.dis.method(self.param_dis_domb, self.gradParam_dis_domb, optimizer.dis.config.lr,
                                optimizer.dis.config.beta1, optimizer.dis.config.beta2,
                                optimizer.dis.config.elipson, optimizer.dis.optimstate.domb)
            optimizer.gen.method(self.param_gen_domab, self.gradParam_gen_domab, optimizer.gen.config.lr,
                                optimizer.gen.config.beta1, optimizer.gen.config.beta2,
                                optimizer.gen.config.elipson, optimizer.gen.optimstate.domab)

            -- save model at every specified epoch.
            local data = {dis_doma = self.dis_doma, gen_domba = self.gen_domba, dis_domb = self.dis_domb, gen_domab = self.gen_domab}
            self:snapshot(string.format('__4_cyclegan/repo/%s', self.opt.name), self.opt.name, totalIter, data)

            -- display server.
            if (totalIter%self.opt.display_iter==0) and (self.opt.display) then
                imgs = {self.x_doma, self.x_doma_tilde, self.x_domb, self.x_domb_tilde}
                self.disp.image(self.x_doma:clone(), 
                                {win=self.opt.display_id + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(self.x_doma_tilde:clone(), 
                                {win=self.opt.display_id*2 + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(self.x_domb:clone(), 
                                {win=self.opt.display_id*4 + self.opt.gpuid, title=self.opt.server_name})
                self.disp.image(self.x_domb_tilde:clone(), 
                                {win=self.opt.display_id*6 + self.opt.gpuid, title=self.opt.server_name})
                
                            -- uncomment this when save png.
                -- save image as png (size 64x64, grid 8x8 fixed).
                local im_png = torch.Tensor(3, self.sampleSize*8, self.sampleSize*8):zero()
                for i = 1, 8 do
                    for j =  1, 8 do
                        im_png[{{},{self.sampleSize*(j-1)+1, self.sampleSize*(j)},{self.sampleSize*(i-1)+1, self.sampleSize*(i)}}]:copy(self.x_tilde[{{8*(i-1)+j},{},{},{}}]:clone():add(1):div(2))
                    end
                end
                os.execute('mkdir -p __4_cyclegan/repo/image')
                image.save(string.format('__4_cyclegan/repo/image/%d.jpg', totalIter), im_png)
            
            
            end

            -- logging
            local log_msg = string.format('Epoch: [%d][%6d/%6d]\tLoss_D(a): %.4f\tLoss_D(b): %.4f\tLoss_G(adv): %.4f\tLoss_G(cycle): %.4f', e, iter, iter_per_epoch, err_dis_doma, err_dis_domb, err_gen_adv, err_gen_cycle)
            print(log_msg)
        end
    end
end


function CycleGAN:snapshot(path, fname, iter, data)
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


return CycleGAN




