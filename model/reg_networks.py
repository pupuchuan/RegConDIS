import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from model.reg_modelio import LoadableModel, store_config_args
from . import networks

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

class RegUnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, norm_layer=nn.InstanceNorm3d, upsample_type='nearest', skip_connection=True):
        super(RegUnet, self).__init__()

        bias_setting = False
        self.skip_connection = skip_connection
        activation = nn.LeakyReLU(0.2)
        # activation = nn.Tanh()  # nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # norm_layer is different to tf
        conv0 = [nn.Conv3d(input_nc, ngf, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=bias_setting), norm_layer(ngf),
                 activation]
        self.conv0 = nn.Sequential(*conv0)

        ### downsample
        mult = 1
        conv_down1 = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                                bias=bias_setting),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down1 = nn.Sequential(*conv_down1)

        mult = 2
        conv_down2 = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                                bias=bias_setting),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down2 = nn.Sequential(*conv_down2)

        mult = 4
        conv_down3 = [
            nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=(1, 2, 2), padding=1, bias=bias_setting),
            norm_layer(ngf * mult * 2), activation]
        self.conv_down3 = nn.Sequential(*conv_down3)

        ### upsample
        mult = 8
        convt_up3 = [nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up3 = nn.Sequential(*convt_up3)

        mult = 4
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv3 = [nn.Conv3d(in_channels, ngf * mult, kernel_size=3, stride=1, padding=1, bias=bias_setting),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv3 = nn.Sequential(*decoder_conv3)

        mult = 4
        convt_up2 = [nn.Upsample(scale_factor=(1, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up2 = nn.Sequential(*convt_up2)

        mult = 2
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv2 = [nn.Conv3d(in_channels, ngf * mult, kernel_size=5, stride=1, padding=2, bias=bias_setting),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv2 = nn.Sequential(*decoder_conv2)

        mult = 2
        convt_up1 = [nn.Upsample(scale_factor=(2, 2, 2), mode=upsample_type),
                     nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,
                               bias=bias_setting),
                     norm_layer(int(ngf * mult / 2)), activation]
        self.convt_up1 = nn.Sequential(*convt_up1)

        if skip_connection:
            in_channels = ngf * 2
        else:
            in_channels = ngf
        decoder_conv1 = [nn.Conv3d(in_channels, output_nc, kernel_size=3, stride=1, padding=1, bias=True), nn.Tanh()]
        self.decoder_conv1 = nn.Sequential(*decoder_conv1)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv_down1(x0)
        x2 = self.conv_down2(x1)
        x3 = self.conv_down3(x2)

        x4 = self.convt_up3(x3)
        if self.skip_connection:
            x4 = torch.cat((x4, x2), dim=1)  # batchsize*channnel*z*x*y
        x4 = self.decoder_conv3(x4)

        x5 = self.convt_up2(x4)
        if self.skip_connection:
            x5 = torch.cat((x5, x1), dim=1)
        x5 = self.decoder_conv2(x5)

        x6 = self.convt_up1(x5)
        if self.skip_connection:
            x6 = torch.cat((x6, x0), dim=1)
        out = self.decoder_conv1(x6)

        return out

class Reg(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        opt,
        inshape,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        # self.unet_model = Unet(
        #     inshape,
        #     nb_features=nb_unet_features,
        #     nb_levels=nb_unet_levels,
        #     feat_mult=unet_feat_mult
        # )

        self.unet_model = RegUnet(2, opt.ngf, opt.ngf)

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(opt.ngf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        # return non-integrated flow field if training
        if not registration:
            return preint_flow, pos_flow
        else:
            return pos_flow

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

