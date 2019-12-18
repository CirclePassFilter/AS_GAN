from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as ta
#import inception_score as tfis
from metric import make_dataset
import numpy as np
# from tensorboardX import SummaryWriter
import random
import model.resnet64_same_sn as model

import torch.nn.functional as F


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

index=0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='lsun', help='cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', default='/data/lfq/', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='EXP/adv_resnet64_sn_p1_lsun', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--advr', type=float, default=1, help='attacking strength, default=1')
    parser.add_argument('--advf', type=float, default=1, help='attacking strength, default=1')
    parser.add_argument('--adv_mode', type=str, default='Dr',choices=['Dr','Df','Drf'],help='attacking mode,choice:[Dr|Df|Drf]')


    ########################################################
    #### For evaluation ####
    parser.add_argument('--sampleSize', type=int, default=20000, help='number of samples for evaluation')
    ########################################################

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    #########################
    #### Dataset prepare ####
    #########################
    dataset = make_dataset(dataset=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    #########################
    #### Models building ####
    #########################
    #torch.cuda.set_device(3)
    #device = torch.device("cuda:3" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3


    netG = model.Generator().cuda()
    #netG.apply(weights_init)
    #netG = nn.DataParallel(netG, device_ids=range(4))
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = model.Discriminator().cuda()
    #netD.apply(weights_init)
    #netD = nn.DataParallel(netD, device_ids=range(4))
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)
    

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(opt.beta1, 0.999))

    # [emd-mmd-knn(knn,real,fake,precision,recall)]*4 - IS - mode_score - FID
    score_tr = np.zeros((opt.niter, 4*7+3))
    incep = np.zeros((opt.niter+1, 2))
    fid = np.zeros((opt.niter+1, 1))

    LOSS = [[] for i in range(3)]   # for D, D_adv, G
    DX = [[] for i in range(2)]     # for D(X),D(X_adv)
    DGZ = [[] for i in range(3)]    # for D(G(Z1)),D(GZ1_adv),D(GZ2)
    def sample_True(save_path,batch_size,batch_num,dataloader):
        image=[]
        print('sampling true###########')
        for i, data in enumerate(dataloader):
            print(i)
            image.append(data[0].numpy())
            batch_num=batch_num-1
            if batch_num==0:
                break
        image=np.array(image)
        image=image.reshape(-1,3,opt.imageSize,opt.imageSize)
        image=image.transpose(0,2,3,1)
        image=(image+1)/2
        print('real',image.shape)
        np.save(save_path,image)
    def sample_fake(save_path,batch_size,batch_num):
            result=[]
            print('sampling fake##########')
            for i in range(batch_num):
                noise = torch.randn(batch_size, nz, 1, 1).cuda()
                image=netG(noise).detach().cpu().numpy()
                image = (image + 1) / 2
                result.append(image)
            result=np.array(result).reshape(-1,3,opt.imageSize,opt.imageSize)
            result=result.transpose(0,2,3,1)
            print('fake',result.shape)
            np.save(save_path,result)




    def check_folder(log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    sample_True(opt.outf+'/real.npy',None,350,dataloader)
    for epoch in range(opt.niter):   ################################################
        #sample_fake(opt.outf+'/epoch_{}.npy'.format(epoch),64,350)
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].cuda()
            real_cpu = real_cpu.requires_grad_(True)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label).cuda()
            real_label_adv = torch.full((batch_size,), real_label).cuda()

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            Grad_Dr = real_cpu.grad
            # def get_pert(errD_real):
            #     pert=0.6-min(errD_real,0.6)
            #     return pert*2.5
            if(i<=300):
                p=0
            else:
                p=1
            x_adv = real_cpu + torch.sign(real_cpu.grad)*p/255.

            x_adv = x_adv.detach()

            # train with fake
            errD=torch.zeros(1).cuda()
            D_G_z1 = torch.zeros(1).cuda()
            noise = torch.randn(batch_size, nz, 1, 1).cuda()
            fake = netG(noise).detach()
            fake = fake.requires_grad_(True)
            label.fill_(fake_label)
            fake_label_adv = label
            output = netD(fake)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            Grad_Df = fake.grad

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            
            x_adv = x_adv.requires_grad_(True)
            optimizerD.zero_grad()
            output_r_adv = netD(x_adv)


            loss_r = criterion(output_r_adv, real_label_adv)
            loss = loss_r


            loss.backward()
            Grad_Dr_adv = x_adv.grad
            D_x_adv = output_r_adv.mean().item()
            optimizerD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            noise = torch.randn(batch_size, nz, 1, 1).cuda()
            fake = netG(noise)
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f | Loss_adv: %.4f | Loss_G: %.4f | D(x): %.4f | D(x_adv): %.4f | D(Gz1,Gz1_adv,Gz2):%.4f/%.4f/%.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         errD.item(), loss.item(), errG.item(), D_x, D_x_adv, D_G_z1, 0, D_G_z2))
                LOSS[0].append(errD.item())
                LOSS[1].append(loss.item())
                LOSS[2].append(errG.item())

                DX[0].append(D_x)
                DX[1].append(D_x_adv)

                DGZ[0].append(D_G_z1)
                DGZ[2].append(D_G_z2)

                ite = epoch * len(dataloader) + i


            if i % 400 == 0:
                torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, index))
                torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, index))
                index=index+1
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)

        # do checkpointing
        


    print('##### training completed :) #####')
    print('### metric scores output is scored at %s/score_tr_ep.npy ###' % opt.outf)
