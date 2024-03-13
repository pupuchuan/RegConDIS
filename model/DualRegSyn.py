from model import networks
import torch
import torch.nn as nn
from reg_losses import *
from model.reg_networks import Reg, SpatialTransformer

class DualRegSyn(nn.Module):
    def __init__(self, opt, SR_weight=20.0, SM_weight=10.0):
        super(DualRegSyn, self).__init__()
        self.isTrain = opt.isTrain
        self.shape = [opt.depthSize, opt.ImageSize, opt.ImageSize]
        self.gpu_ids = opt.gpu_ids
        self.device = opt.device
        self.lambda_L1 = opt.lambda_L1
        self.G_model = opt.G_model
        self.SR_weight = SR_weight
        self.SM_weight = SM_weight
        self.loss_names = ['G_GAN_1', 'SR_1', 'SM_1', 'G_GAN_2', 'SR_2', 'SM_2', 'D_real', 'D_fake', 'D_loss']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, self.shape, opt.G_model, opt.G_norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netR = Reg(opt, self.shape, int_downsize=2).to(self.device)
        # configure transformer
        self.SpatialTransformer = SpatialTransformer(self.shape).to(self.device)
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.D_model,
                                          opt.n_layers_D, opt.D_norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss().to(self.device)
        self.smoothing_loss = Grad('l2', loss_mult=2).loss
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_max, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_max, betas=(opt.beta1, 0.999))
        self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr_max, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.mask = input['mask'].to(self.device)

    def forward(self):

        self.optimizer_R.zero_grad()
        self.optimizer_G.zero_grad()

        Trans, full_Trans = self.netR(self.real_A, self.real_B)

        self.Trans_A_1 = self.SpatialTransformer(self.real_A, full_Trans)
        self.fake_B_1 = self.netG(self.Trans_A_1)
        self.loss_SR_1 = self.SR_weight * self.criterionL1(self.fake_B_1, self.real_B)  #SR
        self.loss_SM_1 = self.SM_weight * self.smoothing_loss(Trans)
        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)
        pred_fake_1 = self.netD(fake_AB_1, isDetach=False)
        self.loss_G_GAN_1 = self.criterionGAN(pred_fake_1, True)

        self.fake_B_2 = self.netG(self.real_A)
        self.Trans_A_2 = self.SpatialTransformer(self.fake_B_2, full_Trans)
        self.loss_SR_2 = self.SR_weight * self.criterionL1(self.Trans_A_2, self.real_B)  #SR
        self.loss_SM_2 = self.SM_weight * self.smoothing_loss(Trans)
        fake_AB_2 = torch.cat((self.real_A, self.Trans_A_2), 1)
        pred_fake_2 = self.netD(fake_AB_2, isDetach=False)
        self.loss_G_GAN_2 = self.criterionGAN(pred_fake_2, True)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN_1 + self.loss_SR_1 + self.loss_SM_1 + \
                      self.loss_G_GAN_2 + self.loss_SR_2 + self.loss_SM_2

        self.loss_G.backward()
        self.optimizer_G.step()  # udpate G's weights
        self.optimizer_R.step()  # udpate G's weightss

        self.optimizer_D.zero_grad()  # set D's gradients to zero
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)

        # calculate gradients for D
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach(), isDetach=True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)  # self.real_A.detach() !!!
        pred_real = self.netD(real_AB, isDetach=True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D_loss = (self.loss_D_fake + self.loss_D_real) / 2
        self.loss_D_loss.backward()
        self.optimizer_D.step()  # update D's weight
