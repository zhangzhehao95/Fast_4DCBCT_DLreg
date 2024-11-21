import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from .blocks import CascadeConv, DownBlock, UpBlock, ResBlock, DownBlock_residual, UpBlock_residual


class UNet(nn.Module):
    def __init__(self, dim, in_channels, cf):
        super().__init__()
        assert dim in (2, 3), 'Only support 2 or 3 dimension.'

        convs_per_level = cf.convs_per_level
        # Customized channel number for each level, where enc_filters[-1] is the bottom level
        enc_filters = cf.enc_filters
        dec_filters = cf. dec_filters
        self.unet_level = len(enc_filters)  # len(enc_filters) = num of unet level = num of up/down + 1

        # Normalization
        if cf.normalization == 'Instance':
            conv_mode = 'CIL'
        elif cf.normalization == 'Batch':
            conv_mode = 'CBL'
        else:
            conv_mode = 'CL'

        # Down/up sampling method
        down_method = cf.downsampling
        up_method = cf.upsampling

        # Down sampling / Encoder
        self.encoder = nn.ModuleList()
        cur_ch = enc_filters[0]
        self.encoder.append(CascadeConv(dim, in_channels, cur_ch, convs_per_level, mode=conv_mode))
        pre_ch = cur_ch
        for layer in range(1, self.unet_level):
            cur_ch = enc_filters[layer]
            # Downsampling and then conv
            self.encoder.append(DownBlock(dim, pre_ch, cur_ch, down=down_method, nb_conv=convs_per_level,
                                          mode=conv_mode))
            pre_ch = cur_ch

        # Up sampling / Decoder
        self.decoder = nn.ModuleList()
        for layer in range(self.unet_level - 1):
            cur_ch = dec_filters[layer]
            # Upsampling, skip connection and then conv
            self.decoder.append(UpBlock(dim, [pre_ch, enc_filters[-2-layer]], cur_ch, up=up_method,
                                        nb_conv=convs_per_level, mode=conv_mode))
            pre_ch = cur_ch

        # Remaining convolutions, out_channels = dec_filters[-1]
        self.remaining = nn.ModuleList()
        for layer in range(len(dec_filters) - self.unet_level + 1):
            cur_ch = dec_filters[self.unet_level - 1 + layer]
            self.remaining.append(CascadeConv(dim, pre_ch, cur_ch, nb_conv=1, mode=conv_mode))
            pre_ch = cur_ch

        # Generate output, (1,dec_filters[-1],d,h,w) --> (1,group*dim,d,h,w)
        final_conv = getattr(nn, 'Conv%dd' % dim)
        self.disp_gen = final_conv(pre_ch, dim * cf.phase_num, kernel_size=1)

    def forward(self, x):
        # x is scaled input (if scaled) rather than the original input
        skips = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            if i < self.unet_level-1:
                skips.append(x)

        for i, up in enumerate(self.decoder):
            x = up(x, skips[-1-i])

        for remain in self.remaining:
            x = remain(x)

        # estimate the displacement field
        x = self.disp_gen(x)
        return x


# 2D or 3D spatial transformer layer to calculate the warped moving image
# Obtained from https://github.com/voxelmorph/voxelmorph
class SpatialTransformer(nn.Module):
    def __init__(self, shape, mode='bilinear'):
        """
        Args:
            shape: size of input to the spatial transformer block (shape of image volume), (h,w) or (d,h,w)
            mode: method of interpolation for grid_sampler. If mode='bilinear' and the input is 5-D, the interpolation
            mode used will actually be trilinear.
        """
        # Called when instantiating the block
        super().__init__()
        self.mode = mode

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)     # Default indexing='ij'
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)     # (1,3,d,h,w)

        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        """
            src (moving image): (n, 1, h, w) or (n, 1, d, h, w)
            flow: (n, 2, h, w) or (n, 3, d, h, w), where n is batch or group_number for group registration
        """
        # New locations
        new_locs = self.grid + flow  # Broadcast self.grid to the same batch number with flow and then plus
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)  # (batch, d, h, w, 3)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, padding_mode='border', align_corners=True)


class ResizeTransform(nn.Module):
    """
    Resize a transform field (DVF), which involves resizing the vector field and rescaling it.
    """
    def __init__(self, size, factor, dim):
        super().__init__()
        # size decides the final output size, factor decides the weights
        self.size = size
        self.factor = factor

        self.mode = 'linear'
        if dim == 2:
            self.mode = 'bi' + self.mode
        elif dim == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # Down-sample, resize first to save memory
            x = F.interpolate(x, size=self.size, align_corners=True, mode=self.mode)    # scale_factor=self.factor
            x = self.factor * x

        elif self.factor > 1:
            # Up-sample, multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, size=self.size, align_corners=True, mode=self.mode)

        # No need to do anything if resize is 1
        return x


class ProcessDisp:
    def __init__(self, disp_shape, calc_device='cuda'):
        """
        Args:
            disp_shape: shape of displacement volume, (d,h,w)
        """
        # https://github.com/vincentme/GroupRegNet
        self.spatial_transform = SpatialTransformer(disp_shape, 'bilinear').to(device=calc_device)
        self.dim = len(disp_shape)
        self.device = torch.device(calc_device)

    def inverse_disp(self, disp, threshold=0.01, max_iteration=20):
        '''
        Compute the inverse field. Implementation of "A simple fixed‐point approach to invert a deformation field"
        Algorithm:
            v_0(x) = 0
            v_n(x) = -u(x + v_n-1(x))
            where u is deformation field and v is the inverse.

        Args:
            disp (displacement field, u): (n, 3, d, h, w) or (3, d, h, w)
        '''
        forward_disp = disp.detach().to(device=self.device)
        if disp.ndim < self.dim + 2:
            forward_disp = torch.unsqueeze(forward_disp, 0)

        backward_disp = torch.zeros_like(forward_disp)
        backward_disp_old = backward_disp.clone()
        for _ in range(max_iteration):
            backward_disp = -self.spatial_transform(forward_disp, backward_disp)
            diff = torch.max(torch.abs(backward_disp - backward_disp_old)).item()
            if diff < threshold:
                break
            backward_disp_old = backward_disp.clone()

        if disp.ndim < self.dim + 2:
            backward_disp = torch.squeeze(backward_disp, 0)
        return backward_disp

    def inverse_disp_v2(self, disp):
        '''
        A coarse-to-fine deformable transformation framework for unsupervised multi-contrast MR image registration with
        dual consistency constraint

        Args:
            disp (displacement field, u): (n, 3, d, h, w) or (3, d, h, w)
        '''
        forward_disp = disp.detach().to(device=self.device)
        if disp.ndim < self.dim + 2:
            forward_disp = torch.unsqueeze(forward_disp, 0)

        backward_disp = -1 * self.spatial_transform(forward_disp, forward_disp)
        return backward_disp

    def compose_disp(self, trans_A, trans_B, mode='corr'):
        '''
        Compute the composition field of A and B, which should be T = A(B(x)). The order matters.

        Args:
            trans_A: displacement field for the first transform, deform phase image to template, f: (m, 3, d, h, w)
            trans_B: displacement field for the second transform, deform template to phase image, g: (n, 3, d, h, w)
            mode: calculation mode, string with default 'corr'
                'corr': generate composition of corresponding displacement field in the batch dimension only, the result
                        shape is the same as input (m/n, 3, d, h, w)
                'all': generate all pairs of composition displacement field. The result shape is (m, n, 3, d, h, w)

        Returns:
            composed_disp:
                'corr': composed_disp[i] gives the DVF for deforming trans_A[i] to trans_B[i]
                'all': composed_disp[i, j] gives the DVF for deforming i to j
        '''
        trans_A_t = trans_A.detach().to(device=self.device)
        trans_B_t = trans_B.detach().to(device=self.device)
        if trans_A.ndim < self.dim + 2:
            trans_A_t = torch.unsqueeze(trans_A_t, 0)
        if trans_B.ndim < self.dim + 2:
            trans_B_t = torch.unsqueeze(trans_B_t, 0)

        if mode == 'corr':
            assert trans_A_t.shape[0] == trans_B_t.shape[0]
            # b + a(x+b(x))
            composed_disp = self.spatial_transform(trans_A_t, trans_B_t) + trans_B_t  # (n, 3, d, h, w)

        elif mode == 'all':
            assert len(trans_A_t.shape) == len(trans_B_t.shape)
            m, _, *image_shape = trans_A.shape
            n = trans_B.shape[0]
            trans_A_mxn = torch.repeat_interleave(torch.unsqueeze(trans_A_t, 1), n, 1)
            trans_A_mn = trans_A_mxn.reshape(m * n, self.dim, *image_shape)

            trans_B_mXn = torch.repeat_interleave(torch.unsqueeze(trans_B_t, 0), m, 0)
            trans_B_mn = trans_B_mXn.reshape(m * n, self.dim, *image_shape)

            composed_disp = self.spatial_transform(trans_A_mn, trans_B_mn).reshape(m, n, self.dim, *image_shape) + trans_B_mXn

        else:
            raise NotImplementedError

        if trans_A.ndim < self.dim + 2 and trans_B.ndim < self.dim + 2:
            composed_disp = torch.squeeze(composed_disp)
        return composed_disp


# UNet with residual block
# https://medium.com/@nishanksingla/unet-with-resblock-for-semantic-segmentation-dd1766b4ff66
class ResUNet(nn.Module):
    def __init__(self, dim, in_channels, cf):
        super().__init__()
        assert dim in (2, 3), 'Only support 2 or 3 dimension.'
        assert cf.convs_per_level == 2, 'Must include two convs in one residual block.'

        # Customized channel number for each level, where enc_filters[-1] is the bottom level
        enc_filters = cf.enc_filters
        dec_filters = cf. dec_filters
        self.unet_level = len(enc_filters)  # len(enc_filters) = num of unet level = num of up/down + 1

        # Down/up sampling method
        down_method = cf.downsampling
        up_method = cf.upsampling

        # Down sampling / Encoder
        self.encoder = nn.ModuleList()
        cur_ch = enc_filters[0]
        self.encoder.append(ResBlock(dim, in_channels, cur_ch))
        pre_ch = cur_ch
        for layer in range(1, self.unet_level):
            cur_ch = enc_filters[layer]
            # Downsampling and then conv
            self.encoder.append(DownBlock_residual(dim, pre_ch, cur_ch, down=down_method))
            pre_ch = cur_ch

        # Up sampling / Decoder
        self.decoder = nn.ModuleList()
        for layer in range(self.unet_level - 1):
            cur_ch = dec_filters[layer]
            # Upsampling, skip connection and then conv
            self.decoder.append(UpBlock_residual(dim, [pre_ch, enc_filters[-2 - layer]], cur_ch, up=up_method))
            pre_ch = cur_ch

        # Remaining convolutions, out_channels = dec_filters[-1]
        self.remaining = nn.ModuleList()
        for layer in range(len(dec_filters) - self.unet_level + 1):
            cur_ch = dec_filters[self.unet_level - 1 + layer]
            self.remaining.append(CascadeConv(dim, pre_ch, cur_ch, nb_conv=1, mode='CIL'))
            pre_ch = cur_ch

        # Generate output, (1,dec_filters[-1],d,h,w) --> (1,group*dim,d,h,w)
        final_conv = getattr(nn, 'Conv%dd' % dim)
        self.disp_gen = final_conv(pre_ch, dim * cf.phase_num, kernel_size=1)

    def forward(self, x):
        # x is scaled input (if scaled) rather than the original input
        skips = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            if i < self.unet_level - 1:
                skips.append(x)

        for i, up in enumerate(self.decoder):
            x = up(x, skips[-1 - i])

        for remain in self.remaining:
            x = remain(x)

        # estimate the displacement field
        x = self.disp_gen(x)
        return x

