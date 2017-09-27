-- For training loop and learning rate scheduling.
-- DiscoGAN.
-- last modified : 2017.09.14, nashory


require 'sys'
require 'optim'
require 'image'
require 'math'
local optimizer = require '__3_discogan.script.optimizer'


local DiscoGAN = torch.class('DiscoGAN')


function DiscoGAN:__init(model, criterion, opt, optimstate)
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
    self.BCEcrit = criterion[1]:cuda()
    self.MSEcrit = criterion[2]:cuda()
end


DiscoGAN['fDx_doma'] = function(self)
    self.dis_doma:zeroGradParameters()
    self.x_doma = self.dataset:getBatchByClass(1)
    self.x_domb = self.dataset:getBatchByClass(2)
    
    -- train with real(x_doma)
    self.label:fill(1)-- real label (1)
    local f_x = self.dis_doma:forward(self.x_doma:cuda()):clone()
    local errD_real = self.BCEcrit:forward(f_x:cuda(), self.label:cuda())
    local d_real_crit = self.BCEcrit:backward(f_x:cuda(), self.label:cuda())
    local d_real_dis = self.dis_doma:backward(self.x_doma:cuda(), d_real_crit:cuda())

    -- train with fake (x_doma_tilde)
    self.label:fill(0)-- fake label (0)
    self.x_doma_tilde = self.gen_domba:forward(self.x_domb:cuda()):clone()
    self.f_x_doma_tilde = self.dis_doma:forward(self.x_doma_tilde:cuda()):clone()
    local errD_fake = self.BCEcrit:forward(self.f_x_doma_tilde:cuda(), self.label:cuda())
    local d_fake_crit = self.BCEcrit:backward(self.f_x_doma_tilde:cuda(), self.label:cuda())
    local d_fake_dis = self.dis_doma:backward(self.x_doma_tilde:cuda(), d_fake_crit:cuda())

    -- return error.
    local errD = errD_real + errD_fake
    return errD
end


DiscoGAN['fGx_domab'] = function(self)
    self.gen_domab:zeroGradParameters()

    -- train with adversrial (L_adv)
    self.label:fill(1)          -- fake labels are real for generator cost.
    local errG_adv = self.BCEcrit:forward(self.f_x_domb_tilde:cuda(), self.label:cuda())
    local d_gen_crit = self.BCEcrit:backward(self.f_x_domb_tilde:cuda(), self.label:cuda())
    local d_gen_dis = self.dis_domb:updateGradInput(self.x_domb_tilde:cuda(), d_gen_crit:cuda())
    local d_gen_dummy = self.gen_domab:backward(self.x_doma:cuda(), d_gen_dis:cuda())
    
    -- train with reconstruction (L_const)
    self.x_domb_const = self.gen_domab(self.x_doma_tilde:cuda())
    local errG_const = self.MSEcrit:forward(self.x_domb_const:cuda(), self.x_domb:cuda())
    d_gen_crit = self.MSEcrit:backward(self.x_domb_const:cuda(), self.x_domb:cuda())
    d_gen_dummy = self.gen_domab:backward(self.x_doma_tilde:cuda(), d_gen_crit:mul(-1))

    local errG = errG_adv + errG_const
    return errG
end

DiscoGAN['fDx_domb'] = function(self)
    self.dis_domb:zeroGradParameters()

    -- train with real(x_domb)
    self.label:fill(1)      -- real label (1)
    local f_x = self.dis_domb:forward(self.x_domb:cuda()):clone()
    local errD_real = self.BCEcrit:forward(f_x:cuda(), self.label:cuda())
    local d_real_crit = self.BCEcrit:backward(f_x:cuda(), self.label:cuda())
    local d_real_dis = self.dis_domb:backward(self.x_domb:cuda(), d_real_crit:cuda())

    -- train with fake(x_doma_tilde)
    self.label:fill(0)      -- fake label (0)
    self.x_domb_tilde = self.gen_domab:forward(self.x_doma:cuda()):clone()
    self.f_x_domb_tilde = self.dis_domb:forward(self.x_domb_tilde:cuda()):clone()
    local errD_fake = self.BCEcrit:forward(self.f_x_domb_tilde:cuda(), self.label:cuda())
    local d_fake_crit = self.BCEcrit:backward(self.f_x_domb_tilde:cuda(), self.label:cuda())
    local d_fake_dis = self.dis_domb:backward(self.x_domb_tilde:cuda(), d_fake_crit:cuda())

    -- return error.
    local errD = errD_real + errD_fake
    return errD
end

DiscoGAN['fGx_domba'] = function(self)
    self.gen_domba:zeroGradParameters()

    -- train with adversrial (L_adv)
    self.label:fill(1)          -- fake labels are real for generator cost.
    local errG_adv = self.BCEcrit:forward(self.f_x_doma_tilde:cuda(), self.label:cuda())
    local d_gen_crit = self.BCEcrit:backward(self.f_x_doma_tilde:cuda(), self.label:cuda())
    local d_gen_dis = self.dis_doma:updateGradInput(self.x_doma_tilde:cuda(), d_gen_crit:cuda())
    local d_gen_dummy = self.gen_domba:backward(self.x_domb:cuda(), d_gen_dis:cuda())
    
    -- train with reconstruction (L_const)
    self.x_doma_const = self.gen_domba(self.x_domb_tilde:cuda())
    local errG_const = self.MSEcrit:forward(self.x_doma_const:cuda(), self.x_doma:cuda())
    d_gen_crit = self.MSEcrit:backward(self.x_doma_const:cuda(), self.x_doma:cuda())
    d_gen_dummy = self.gen_domba:backward(self.x_domb_tilde:cuda(), d_gen_crit:mul(-1))

    local errG = errG_adv + errG_const
    return errG
end


function DiscoGAN:train(epoch, loader)
    -- Initialize data variables.
    self.label = torch.Tensor(self.batchSize):zero()

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


    -- data for test
    self.test_doma = self.dataset:getBatchByClass(1)
    self.test_domb = self.dataset:getBatchByClass(2)


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
            local err_gen_domab = self:fGx_domab()
            local err_gen_domba = self:fGx_domba()

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
            self:snapshot(string.format('__3_discogan/repo/%s', self.opt.name), self.opt.name, totalIter, data)

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
                local im_png2 = torch.Tensor(3, self.sampleSize*8, self.sampleSize*8):zero()
                local im_domb = self.gen_domab:forward(self.test_doma:cuda())
                local im_doma = self.gen_domba:forward(self.test_domb:cuda())
                for i = 1, 8 do
                    for j =  1, 8 do
                        im_png[{{},{self.sampleSize*(j-1)+1, self.sampleSize*(j)},{self.sampleSize*(i-1)+1, self.sampleSize*(i)}}]:copy(im_doma[{{8*(i-1)+j},{},{},{}}]:clone():add(1):div(2))
                        im_png2[{{},{self.sampleSize*(j-1)+1, self.sampleSize*(j)},{self.sampleSize*(i-1)+1, self.sampleSize*(i)}}]:copy(im_domb[{{8*(i-1)+j},{},{},{}}]:clone():add(1):div(2))
                    end
                end
                os.execute('mkdir -p __3_discogan/repo/image/doma')
                os.execute('mkdir -p __3_discogan/repo/image/domb')
                image.save(string.format('__3_discogan/repo/image/doma/%d.jpg', totalIter/self.opt.display_iter), im_png)
                image.save(string.format('__3_discogan/repo/image/domb/%d.jpg', totalIter/self.opt.display_iter), im_png2)
            
            
            end

            -- logging
            local log_msg = string.format('Epoch: [%d][%6d/%6d]\tLoss_D(a): %.4f\tLoss_G(ba): %.4f\tLoss_D(b): %.4f\tLoss_G(ab): %.4f', e, iter, iter_per_epoch, err_dis_doma, err_gen_domba, err_dis_domb, err_gen_domab)
            print(log_msg)
        end
    end
end


function DiscoGAN:snapshot(path, fname, iter, data)
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


return DiscoGAN




