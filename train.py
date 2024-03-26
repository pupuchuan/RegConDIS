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

parser.add_argument('--gpu', type=str, default='6', help='which gpu is used')
parser.add_argument('--G_model', type=str, default='Unet', help='specify generator architecture [Unet | global | GLFANET | swin_trans ]')
parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
parser.add_argument('--depthSize', type=int, default=8, help='depth for 3d images')
parser.add_argument('--ImageSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--Npatch', type=int, default=10, help='Npatch')
parser.add_argument('--trainer', type=str, default='DualRegSyn', help='pix2pix | CycleGAN | ADN | UNIT | MUINT | RegGAN | DIS | UNIT | MUNIT....')
parser.add_argument('--max_epochs', type=int, default=200, help='# max_epoch')
parser.add_argument('--seed', type=int, default=15, help='random seed')
parser.add_argument('--lr_max', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--gan_mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan ｜ wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument("--image_dir", type=str, default='', help="image_dir")
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
parser.add_argument('--load_name', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--CT_max', type=int, default=1500, help='CT_max')
parser.add_argument('--CT_min', type=int, default=-1000, help='CT_min')

parser.add_argument('--isTrain', action='store_false', help='isTrain')
parser.add_argument('--print_freq_num', type=int, default=12, help='frequency of showing training results on console')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--disx', type=int, default=10120, help='frequency of showing training results on console')
opt = parser.parse_args()

opt.code_dir = code_dir
opt.checkpoints_name = '{}/{}_{}/Npatch{}_xyz_{}_{}_epoch_{}_bs_{}_nlayersD_{}_ctmin_{}_ctmax_{}_seed_{}' .format(opt.trainer,
    opt.G_model, opt.D_model, opt.Npatch, opt.depthSize, opt.ImageSize, opt.max_epochs, opt.batch_size, opt.n_layers_D,
    opt.CT_min, opt.CT_max, opt.seed)

opt.checkpoints_dir = os.path.join(opt.code_dir, 'checkpoints', opt.checkpoints_name)
opt.model_results = os.path.join(opt.checkpoints_dir, 'model_results')
opt.file_name_txt = os.path.join(opt.checkpoints_dir, 'train_message.txt')
opt.prediction_results = os.path.join(opt.checkpoints_dir, 'prediction_results_')
opt.pretrain_model_path = os.path.join(opt.code_dir, opt.loss_pre_dir)

pathlib.Path(opt.model_results).mkdir(parents=True, exist_ok=True)

print_options(opt)

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

train_set = DatasetFromFolder_train(opt)
train_dataloader = DataLoader(dataset=train_set, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True)

train_size = len(train_dataloader)    # get the number of images in the dataset.
print('The number of train images = %d' % train_size)

opt.print_freq = int(train_size/opt.print_freq_num)

if 'DualRegSyn' == opt.trainer:
    model = DualRegSyn(opt)
else:
    raise ValueError('trainer does not exist! please check opt.trainer!')

train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/train'))
val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'log/val'), flush_secs=2)

best_MAE = 500
total_iters = 0
save_nii_epoch = [i for i in range(0, opt.max_epochs + 1, 20)]
save_network_epoch = [i for i in range(50, opt.max_epochs, 10)]
print('training')

val_image_filenames = os.listdir(os.path.join(opt.image_dir, 'val'))
val_pf = pd.DataFrame(index=(val_image_filenames + ['Mean']))

for epoch in range(opt.epoch_count, opt.max_epochs + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
    image_num = 0
    epoch_train_MAE = []

    for i, data in enumerate(train_dataloader):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        total_iters += 1
        epoch_iter += 1
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.forward()  # calculate loss functions, get gradients, update network weights

        Real_ct = inverser_norm_ct(model.real_B.detach().cpu().numpy(), opt.CT_max, opt.CT_min)
        Fake_ct = inverser_norm_ct(model.fake_B.detach().cpu().numpy(), opt.CT_max, opt.CT_min)
        ct_mask = model.mask.detach().cpu().numpy()

        MAE = (np.abs(Fake_ct - Real_ct) * ct_mask).sum() / ct_mask.sum()
        epoch_train_MAE.append(MAE)

        if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = get_current_losses(model)
            lr = model.optimizer_G.param_groups[0]['lr']
            print_current_message(epoch, epoch_iter, train_size, lr, losses, MAE)
            current_iter_display = epoch + i / train_size
            train_writer.add_scalar('learning_rate', lr, total_iters)
            for k, v in losses.items():
                train_writer.add_scalar('%s' % k, v, total_iters)

        iter_data_time = time.time()

    update_learning_rate(model, opt.max_epochs, epoch, opt.lr_max, opt.trainer)

    epoch_val_MAE = []
    epoch_val_SSIM = []
    epoch_val_PSNR = []

    patch_size = opt.ImageSize
    patch_deep = opt.depthSize

    Val_run = False
    if epoch < 0:
        Val_run = True
    elif (epoch > 3) & (epoch < int(opt.max_epochs * 0.5)):
        if epoch % 10 == 0:
            Val_run = True
    elif (epoch > int(opt.max_epochs * 0.5)) & (epoch < int(opt.max_epochs * 0.8)):
        if epoch % 5 == 0:
            Val_run = True
    else:
        Val_run = True

    if Val_run:
        with torch.no_grad():
            for sub in range(len(val_image_filenames)):
                this_sub = val_image_filenames[sub]
                source, _, _, _ = NiiDataRead(
                    os.path.join(opt.image_dir, 'val', this_sub, 'Arterial.nii.gz'))
                target, spacing, origin, direction = NiiDataRead(
                    os.path.join(opt.image_dir, 'val', this_sub, 'NC_nonlinear_deformed.nii.gz'))    # 准确的MAE
                MASK, _, _, _ = NiiDataRead(
                    os.path.join(opt.image_dir, 'val', this_sub, 'mask.nii.gz'))

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
                        if opt.trainer == 'CycleGAN' or opt.trainer == 'DIS' or opt.trainer == 'UNIT' or 'DIS' in opt.trainer:
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
                count_used[count_used==0] = 1
                output = output / count_used
                output[MASK == 0] = -1
                output = inverser_norm_ct(output, opt.CT_max, opt.CT_min)

                if epoch in save_nii_epoch:
                    prediction_results_val = os.path.join(opt.prediction_results + 'val_results', 'epoch' + str(epoch))
                    save_pre_path = os.path.join(prediction_results_val, val_image_filenames[sub] + '.nii.gz')
                    pathlib.Path(prediction_results_val).mkdir(parents=True, exist_ok=True)
                    NiiDataWrite(save_pre_path, output, spacing, origin, direction)

                data_range =opt.CT_max - opt.CT_min
                MAE = (np.abs(output - target) * MASK).sum() / MASK.sum()
                SSIM = structural_similarity(output, target, data_range=data_range)
                PSNR = peak_signal_noise_ratio(output[MASK > 0], target[MASK > 0], data_range=data_range)
                epoch_val_MAE.append(MAE)
                epoch_val_SSIM.append(SSIM)
                epoch_val_PSNR.append(PSNR)
                message = 'epoch[%d/%d] val[%d/%d]: The MAE, SSIM, PSNR of %s is %.3f,%.3f,%.3f, the total MAE, SSIM, PSNR is %.3f,%.3f,%.3f' % \
                          (epoch, opt.max_epochs, sub + 1, len(val_image_filenames), this_sub, MAE, SSIM, PSNR, np.mean(epoch_val_MAE),
                           np.mean(epoch_val_SSIM), np.mean(epoch_val_PSNR))
                print(message)
                with open(opt.file_name_txt, 'a') as opt_file:
                    opt_file.write(message)
                    opt_file.write('\n')

        mean_val_MAE = np.mean(epoch_val_MAE)
        mean_val_SSIM = np.mean(epoch_val_SSIM)
        mean_val_PSNR = np.mean(epoch_val_PSNR)

        epoch_val_MAE.append(mean_val_MAE)
        epoch_val_SSIM.append(mean_val_SSIM)
        epoch_val_PSNR.append(mean_val_PSNR)

        val_pf['epoch{}_MAE'.format(epoch)] = epoch_val_MAE
        val_pf['epoch{}_SSIM'.format(epoch)] = epoch_val_SSIM
        val_pf['epoch{}_PSNR'.format(epoch)] = epoch_val_PSNR

        val_pf.to_csv(os.path.join(opt.checkpoints_dir, 'val_result.csv'))

        val_writer.add_scalar('val_MAE', mean_val_MAE, epoch)

        if mean_val_MAE < best_MAE:
            best_MAE = mean_val_MAE
            save_networks(opt, 'best_MAE', model, epoch, opt.trainer)

            val_best_pf = pd.DataFrame(index=(val_image_filenames + ['Mean']))
            val_best_pf['epoch{}_MAE'.format(epoch)] = epoch_val_MAE
            val_best_pf['epoch{}_SSIM'.format(epoch)] = epoch_val_SSIM
            val_best_pf['epoch{}_PSNR'.format(epoch)] = epoch_val_PSNR
            val_best_pf.to_csv(os.path.join(opt.checkpoints_dir, 'val_best_result.csv'))

    print('saving the model')
    save_networks(opt, 'latest', model, epoch, opt.trainer)

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.max_epochs, time.time() - epoch_start_time))

train_writer.close()