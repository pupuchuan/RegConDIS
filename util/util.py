from __future__ import print_function
from collections import OrderedDict
import os
import torch
import numpy as np

def save_networks(opt, save_name, model, epoch, model_name):

    if model_name == 'DualRegDIS':
        save_filename = '%s.pth' % (save_name)
        save_path = os.path.join(opt.model_results, save_filename)

        state = {
            'epoch': epoch + 1,
            'netG_state_dict': model.netG.state_dict(),
            'netR_state_dict': model.netR.state_dict(),
            'netD_A_state_dict': model.netD_A.state_dict(),
            'netD_B_state_dict': model.netD_B.state_dict(),
            'optimizer_G': model.optimizer_G.state_dict(),
            'optimizer_R': model.optimizer_D_A.state_dict(),
            'optimizer_D_A': model.optimizer_D_A.state_dict(),
            'optimizer_D_B': model.optimizer_D_B.state_dict()
        }
        torch.save(state, save_path)

    elif model_name == 'CycleGAN' or model_name == 'UNIT':
        save_filename = '%s.pth' % (save_name)
        save_path = os.path.join(opt.model_results, save_filename)

        state = {
            'epoch': epoch + 1,
            'netG_A2B_state_dict': model.netG_A2B.state_dict(),
            'netG_B2A_state_dict': model.netG_B2A.state_dict(),
            'netD_A_state_dict': model.netD_A.state_dict(),
            'netD_B_state_dict': model.netD_B.state_dict(),
            'optimizer_G': model.optimizer_G.state_dict(),
            'optimizer_D_A': model.optimizer_D_A.state_dict(),
            'optimizer_D_B': model.optimizer_D_B.state_dict()
        }

        torch.save(state, save_path)

    elif 'DIS' in model_name:
        save_filename = '%s.pth' % (save_name)
        save_path = os.path.join(opt.model_results, save_filename)

        state = {
            'epoch': epoch + 1,
            'netG_state_dict': model.netG.state_dict(),
            'netD_A_state_dict': model.netD_A.state_dict(),
            'netD_B_state_dict': model.netD_B.state_dict(),
            'optimizer_G': model.optimizer_G.state_dict(),
            'optimizer_R': model.optimizer_D_A.state_dict(),
            'optimizer_D_A': model.optimizer_D_A.state_dict(),
            'optimizer_D_B': model.optimizer_D_B.state_dict()
        }
        torch.save(state, save_path)

    elif 'Reg' in model_name:
        save_filename = '%s.pth' % (save_name)
        save_path = os.path.join(opt.model_results, save_filename)

        state = {
            'epoch': epoch + 1,
            'netG_state_dict': model.netG.state_dict(),
            'netD_state_dict': model.netD.state_dict(),
            'netR_state_dict': model.netR.state_dict(),
            'optimizer_R': model.optimizer_D_A.state_dict(),
            'optimizer_G': model.optimizer_G.state_dict(),
            'optimizer_D': model.optimizer_D.state_dict()
        }

        torch.save(state, save_path)
    else:
        save_filename = '%s.pth' % (save_name)
        save_path = os.path.join(opt.model_results, save_filename)

        state = {
            'epoch': epoch + 1,
            'netG_state_dict': model.netG.state_dict(),
            'netD_state_dict': model.netD.state_dict(),
            'optimizer_G': model.optimizer_G.state_dict(),
            'optimizer_D': model.optimizer_D.state_dict()
        }

        torch.save(state, save_path)


def load_networks(opt, model):

    load_filename = '%s.pth' % (opt.load_name)
    load_path = os.path.join(opt.model_results, load_filename)

    # if isinstance(net, torch.nn.DataParallel):
    #     net = net.module
    print('loading the model from %s' % load_path)

    # state = torch.load(load_path)
    state = torch.load(load_path, map_location='cpu')

    if opt.trainer == 'CycleGAN' or opt.trainer == 'UNIT':

        pretrained_netG_A2B_dict = state['netG_A2B_state_dict']
        model_netG_A2B_dict = model.netG_A2B.state_dict()
        pretrained_netG_A2B_dict = {k: v for k, v in pretrained_netG_A2B_dict.items() if k in model_netG_A2B_dict}
        model_netG_A2B_dict.update(pretrained_netG_A2B_dict)
        model.netG_A2B.load_state_dict(model_netG_A2B_dict)  # torch.load: 加载训练好的模型 load_state_dict: 将torch.load加载出来的数据加载到net中

        if opt.isTrain:

            pretrained_netG_B2A_dict = state['netG_B2A_state_dict']
            model_netG_B2A_dict = model.netG_B2A.state_dict()
            pretrained_netG_B2A_dict = {k: v for k, v in pretrained_netG_B2A_dict.items() if k in model_netG_B2A_dict}
            model_netG_B2A_dict.update(pretrained_netG_B2A_dict)
            model.netG_B2A.load_state_dict(
                model_netG_B2A_dict)  

            pretrained_netD_dict = state['netD_state_dict']
            model_netD_dict = model.netD.state_dict()
            pretrained_netD_dict = {k: v for k, v in pretrained_netD_dict.items() if k in model_netD_dict}
            model_netD_dict.update(pretrained_netD_dict)
            model.netD.load_state_dict(
                model_netD_dict) 

            model.optimizer_G.load_state_dict(state['optimizer_G'])
            model.optimizer_D_A.load_state_dict(state['optimizer_D_A'])
            model.optimizer_D_B.load_state_dict(state['optimizer_D_B'])

    else:

        pretrained_netG_dict = state['netG_state_dict']
        model_netG_dict = model.netG.state_dict()
        pretrained_netG_dict = {k: v for k, v in pretrained_netG_dict.items() if k in model_netG_dict}
        model_netG_dict.update(pretrained_netG_dict)
        model.netG.load_state_dict(model_netG_dict)  

        if opt.isTrain:
            pretrained_netD_dict = state['netD_state_dict']
            model_netD_dict = model.netD.state_dict()
            pretrained_netD_dict = {k: v for k, v in pretrained_netD_dict.items() if k in model_netD_dict}
            model_netD_dict.update(pretrained_netD_dict)
            model.netD.load_state_dict(
                model_netD_dict)  

            model.optimizer_G.load_state_dict(state['optimizer_G'])
            model.optimizer_D.load_state_dict(state['optimizer_D'])


    opt.epoch_count = state['epoch']
    print('Successfully loading the model from %s' % load_path)

def print_current_message(epoch, iters, dataset_size, lr, losses, MAE):
    message = '(epoch: %d, iters: %d/%d, lr: %.6f, train_MAE: %.3f  )' % (epoch, iters, dataset_size, lr, MAE)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message

def get_current_losses(model):
    """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
    errors_ret = OrderedDict()
    for name in model.loss_names:
        if isinstance(name, str):
            errors_ret[name] = float(getattr(model, 'loss_' + name))  # float(...) works for both scalar tensor and float number
    return errors_ret

def update_learning_rate(model, max_epochs, epoch, lr_max, model_name):
    """Update learning rates for all the networks; called at the end of every epoch"""
    # for scheduler in self.schedulers:
    #     scheduler.step()
    if model_name == 'CycleGAN' or 'DIS' in model_name:
        model.optimizer_G.param_groups[0]['lr'] = lr_max * (1 - epoch / max_epochs) ** 0.9
        model.optimizer_D_A.param_groups[0]['lr'] = model.optimizer_G.param_groups[0]['lr']
        model.optimizer_D_B.param_groups[0]['lr'] = model.optimizer_G.param_groups[0]['lr']
        if 'Reg' in model_name:
            model.optimizer_R.param_groups[0]['lr'] = model.optimizer_G.param_groups[0]['lr']
    else:
        model.optimizer_G.param_groups[0]['lr'] = lr_max * (1 - epoch / max_epochs) ** 0.9
        model.optimizer_D.param_groups[0]['lr'] = model.optimizer_G.param_groups[0]['lr']
        if 'Reg' in model_name:
            model.optimizer_R.param_groups[0]['lr'] = model.optimizer_G.param_groups[0]['lr']

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def inverser_norm_ct(x, max_vax, max_min):
    x = (x+1) / 2
    x = (max_vax - max_min) * x - abs(max_min)
    return x

def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)
    if not opt.continue_train:
        with open(opt.file_name_txt, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

def normalization_mr(X):
    p2, p99 = np.percentile(X, (2, 99))
    X[X < p2] = p2
    X[X > p99] = p99
    result = ((X - p2) / (p99 - p2)).astype('float')
    return result * 2 - 1

def normalization_ct(data, min_value, max_value):
    if type(data) is not np.ndarray:
        data = data.numpy()
    nor_data = (data - min_value)/(max_value - min_value)
    last_data = (nor_data - 0.5)*2
    return last_data