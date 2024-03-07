import time
import pathlib
from torch.utils.data import DataLoader
import pandas as pd
from data.dataset3D import DatasetFromFolder_train
from util.Nii_utils import NiiDataRead, NiiDataWrite
from util.util import *
from tensorboardX import SummaryWriter
from model.DualRegSyn import DualRegSyn
import random
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import os
import argparse

code_dir = os.getcwd()
parser = argparse.ArgumentParser()

# 经常需要修改的接口
parser.add_argument("--image_dir", type=str, default='', help="image_dir")
parser.add_argument('--gpu', type=str, default='3', help='which gpu is used')
parser.add_argument('--G_model', type=str, default='Unet', help='specify generator architecture [global | GLFANET | swin_trans ]')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--depthSize', type=int, default=8, help='depth for 3d images')
parser.add_argument('--Npatch', type=int, default=10, help='Npatch')
parser.add_argument('--ImageSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--trainer', type=str, default='DualRegSyn', help='trainer')
parser.add_argument('--dataset', type=str, default='test', help='train | val | test')
parser.add_argument('--load_name', type=str, default='best_MAE', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--checkpoints_dir', type=str, default='')

# set base_options
parser.add_argument('--gpu_ids', type=str, default='0', help='which gpu is used')
parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--D_model', type=str, default='n_layers', help='specify discriminator architecture [wave3DDiscriminator | n_layers | swinDiscriminator]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=16, help='# of discrim filters in the first conv layer')
parser.add_argument('--n_layers_D', type=int, default=2, help='only used if netD==n_layers')
parser.add_argument('--G_norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--D_norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--lambda_L1', type=float, default=20, help='weight for L1 loss')
parser.add_argument('--seed', type=int, default=15, help='random seed')
parser.add_argument('--CT_max', type=int, default=1500, help='CT_max')
parser.add_argument('--CT_min', type=int, default=-1000, help='CT_min')
parser.add_argument('--VGG_loss', action='store_false', help='isVGG')

# set train_options
parser.add_argument('--isTrain', action='store_true', help='isTrain')
parser.add_argument('--print_freq_num', type=int, default=4, help='frequency of showing training results on console')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--max_epochs', type=int, default=100, help='# max_epoch')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr_max', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--loss_pre_dir', type=str, default='perceive_loss/vgg19-dcbb9e9d.pth', help='resnet18_pretrain_path')
# set val_options
parser.add_argument('--disx', type=int, default=10120, help='frequency of showing training results on console')
opt = parser.parse_args()

opt.prediction_results = os.path.join(opt.checkpoints_dir, f'prediction_results_{opt.dataset}_{opt.load_name}')  #
opt.model_results = os.path.join(opt.checkpoints_dir, 'model_results')

# pathlib.Path(opt.model_results).mkdir(parents=True, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

os.environ['PYTHONHASHSEED'] = opt.seed
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if 'DualRegSyn' == opt.trainer:
    model = DualRegSyn(opt)
else:
    raise ValueError('trainer does not exist! please check opt.trainer!')

if not opt.isTrain or opt.continue_train:
    load_networks(opt, model)

epoch_val_MAE = []
epoch_val_SSIM = []
epoch_val_PSNR = []

patch_size = opt.ImageSize
patch_deep = opt.depthSize

val_image_filenames = os.listdir(os.path.join(opt.image_dir, opt.dataset))
val_pf = pd.DataFrame(index=(val_image_filenames + ['Mean'] + ['Std']))

with torch.no_grad():
    for sub in range(len(val_image_filenames)):
        this_sub = val_image_filenames[sub]
        source, _, _, _ = NiiDataRead(
            os.path.join(opt.image_dir, opt.dataset, this_sub, 'Arterial.nii.gz'))
        target, spacing, origin, direction = NiiDataRead(
            os.path.join(opt.image_dir, opt.dataset, this_sub, 'NC_nonlinear_deformed.nii.gz'))
        MASK, _, _, _ = NiiDataRead(
            os.path.join(opt.image_dir, opt.dataset, this_sub, 'mask.nii.gz'))

        source[source < opt.CT_min] = opt.CT_min
        source[source > opt.CT_max] = opt.CT_max
        source[MASK == 0] = opt.CT_min
        source = normalization_ct(source, opt.CT_min, opt.CT_max)

        z, y, x = np.where((MASK > 0))

        z_edge1 = np.where((z + patch_deep / 2) > source.shape[0])
        z[z_edge1] = source.shape[0] - patch_deep / 2

        z_edge2 = np.where((z - patch_deep / 2) < 0)
        z[z_edge2] = patch_deep / 2

        y_edge1 = np.where((y + patch_size / 2) > source.shape[1])
        y[y_edge1] = source.shape[1] - patch_size / 2

        y_edge2 = np.where((y - patch_size / 2) < 0)
        y[y_edge2] = patch_size / 2

        x_edge1 = np.where((x + patch_size / 2) > source.shape[2])
        x[x_edge1] = source.shape[2] - patch_size / 2

        x_edge2 = np.where((x - patch_size / 2) < 0)
        x[x_edge2] = patch_size / 2

        source = source[None, :, :, :]

        output = np.zeros((MASK.shape[0], MASK.shape[1], MASK.shape[2])).astype('float')
        count_used = np.zeros((MASK.shape[0], MASK.shape[1], MASK.shape[2])).astype('float')
        dis = opt.disx
        for num in range(len(x)):
            if num % dis == 0:
                deep = z[num]
                height = y[num]
                width = x[num]
                X_source = source[:, int(deep - patch_deep / 2):int(deep + patch_deep / 2),
                           int(height - patch_size / 2):int(height + patch_size / 2),
                           int(width - patch_size / 2):int(width + patch_size / 2)]
                X_source = torch.tensor(X_source).unsqueeze(0).float().to(opt.device)
                if opt.trainer == 'CycleGAN' or 'DIS' in opt.suffix:
                    target_pred = model.A2B(X_source)
                else:
                    target_pred = model.netG(X_source)
                target_pred = np.squeeze(target_pred.cpu().numpy())
                target_pred[target_pred < -1] = -1
                target_pred[target_pred > 1] = 1
                output[int(deep - patch_deep / 2):int(deep + patch_deep / 2),
                int(height - patch_size / 2):int(height + patch_size / 2),
                int(width - patch_size / 2):int(width + patch_size / 2)] += target_pred
                count_used[int(deep - patch_deep / 2):int(deep + patch_deep / 2),
                int(height - patch_size / 2):int(height + patch_size / 2),
                int(width - patch_size / 2):int(width + patch_size / 2)] += 1
        count_used[count_used == 0] = 1
        output = output / count_used
        output[MASK == 0] = -1
        output = inverser_norm_ct(output, opt.CT_max, opt.CT_min)

        save_pre_path = os.path.join(opt.prediction_results, val_image_filenames[sub] + '.nii.gz')
        pathlib.Path(opt.prediction_results).mkdir(parents=True, exist_ok=True)
        NiiDataWrite(save_pre_path, output, spacing, origin, direction)

        data_range = opt.CT_max - opt.CT_min
        MAE = (np.abs(output - target) * MASK).sum() / MASK.sum()
        SSIM = structural_similarity(output, target, data_range=data_range)
        PSNR = peak_signal_noise_ratio(output[MASK > 0], target[MASK > 0], data_range=data_range)
        epoch_val_MAE.append(MAE)
        epoch_val_SSIM.append(SSIM)
        epoch_val_PSNR.append(PSNR)
        message = 'test[%d/%d]: The MAE, SSIM, PSNR of %s is %.3f,%.3f,%.3f, the total MAE, SSIM, PSNR is %.3f,%.3f,%.3f' % \
                  (sub + 1, len(val_image_filenames), this_sub, MAE, SSIM, PSNR, np.mean(epoch_val_MAE),
                   np.mean(epoch_val_SSIM), np.mean(epoch_val_PSNR))
        print(message)

mean_val_MAE = np.mean(epoch_val_MAE)
mean_val_SSIM = np.mean(epoch_val_SSIM)
mean_val_PSNR = np.mean(epoch_val_PSNR)

std_val_MAE = np.std(epoch_val_MAE)
std_val_SSIM = np.std(epoch_val_SSIM)
std_val_PSNR = np.std(epoch_val_PSNR)

epoch_val_MAE.append(mean_val_MAE)
epoch_val_SSIM.append(mean_val_SSIM)
epoch_val_PSNR.append(mean_val_PSNR)

epoch_val_MAE.append(std_val_MAE)
epoch_val_SSIM.append(std_val_SSIM)
epoch_val_PSNR.append(std_val_PSNR)

val_pf['{}_MAE'.format(opt.load_name)] = epoch_val_MAE
val_pf['{}_SSIM'.format(opt.load_name)] = epoch_val_SSIM
val_pf['{}_PSNR'.format(opt.load_name)] = epoch_val_PSNR

val_pf.to_csv(os.path.join(opt.prediction_results, 'test_result.csv'))


