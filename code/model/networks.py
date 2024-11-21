import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import UNet, SpatialTransformer, ResizeTransform, ResUNet


# One-to-one phase registration
class RegPair(nn.Module):
    """
    Pairwise registration: one-to-one template registration
    """
    def __init__(self, image_shape, cf):
        """
        Args:
            image_shape: Input spatial dimensions, (d, h, w)
            cf: Configuration
        """
        super().__init__()
        self.dim = len(image_shape)
        # scale < 1: down-sample the input images -> calculate dvf -> up-sample dvf -> apply to original inputs
        # scale should not be larger than 1.
        self.scale = cf.dvf_scale

        self.unet_model = UNet(self.dim, in_channels=2, cf=cf)
        # Init displacement layer with small weights and bias
        Conv = getattr(nn, 'Conv%dd' % self.dim)
        self.disp_gen = Conv(self.unet_model.final_nf, self.dim, kernel_size=3, padding=1)
        self.disp_gen.weight = nn.Parameter(Normal(0, 1e-5).sample(self.disp_gen.weight.shape), requires_grad=True)
        self.disp_gen.bias = nn.Parameter(torch.zeros(self.disp_gen.bias.shape), requires_grad=True)

        self.spatial_transform = SpatialTransformer(image_shape, cf.STN_interpolation)

    def forward(self, src, tar, dvf=True):
        """
        Args:
            src: Source image tensor (moving), (batch, 1, d, h, w)
            tar: Target image tensor (fixed), (batch, 1, d, h, w)
            dvf: Whether return the calculated dvf.

        Returns:
            warped_input: Warped moving image, (batch, 1, d, h, w)
            dvf: displacement, (batch, 3, d, h, w)
        """
        ori_input_shape = src.shape[2:]   # (d,h,w)
        x = torch.cat([src, tar], dim=1)  # (b,2,d,h,w), concatenate along channels

        # Down-sample the input images
        if self.scale < 1:
            scaled_x = F.interpolate(x, scale_factor=self.scale, align_corners=True,
                                     mode='bilinear' if self.dim == 2 else 'trilinear', recompute_scale_factor=False)
        else:
            scaled_x = x

        # Calculate DVFs through Unet
        scaled_disp = self.unet_model(scaled_x)
        scaled_disp = self.disp_gen(scaled_disp)  # (b,dec_filters[-1],d,h,w) --> (b,dim=3,d,h,w)

        # Up-sample DVFs
        if self.scale < 1:
            disp = ResizeTransform(size=ori_input_shape, factor=1.0 / self.scale, dim=self.dim)(scaled_disp)
        else:
            disp = scaled_disp

        # Apply DVFs to src
        # img (batch,1,d,h,w), disp (batch,3,d,h,w) --> (batch,1,d,h,w)
        deformed_image = self.spatial_transform(src, disp)

        res = {'warped_input': deformed_image}
        if dvf:
            res['dvf'] = disp

        return res


# Simultaneous multi-phases registration
class RegGroup(nn.Module):
    """
    Group registration: all-to-one common template registration
    """
    def __init__(self, image_shape, group_num=10, cf=None):
        """
        Args:
            image_shape: Input spatial dimensions, (d, h, w)
            group_num: Phase number
            cf: Configuration
        """
        super().__init__()
        self.dim = len(image_shape)
        self.group = group_num
        # scale < 1: down-sample the input images -> calculate dvf -> up-sample dvf -> apply to original inputs
        # scale should not be larger than 1.
        self.scale = cf.dvf_scale

        if cf.network_type == 'UNet':
            self.disp_gen_model = UNet(self.dim, in_channels=self.group, cf=cf)
        elif cf.network_type == 'ResUNet':
            self.disp_gen_model = ResUNet(self.dim, in_channels=self.group, cf=cf)

        self.spatial_transform = SpatialTransformer(image_shape, cf.STN_interpolation)

    def forward(self, input_images, dvf=True, template=True):
        """
        Args:
            input_images: Group of all phase images, (batch=1, channel=group_num, d, h, w)
            dvf: Whether return the calculated dvf.
            template: Whether return the common implicit template image to which all input are deformed.

        Returns:
            warped_input: Warped moving image, (group_num, 1, d, h, w)
            dvf: displacement, (group_num, 3, d, h, w)
            template: Implicit template image, (1, 1, d, h, w)
        """
        ori_input_shape = input_images.shape[2:]

        # Down-sample the input images
        if self.scale < 1:
            scaled_x = F.interpolate(input_images, scale_factor=self.scale, align_corners=True,
                                     mode='bilinear' if self.dim == 2 else 'trilinear', recompute_scale_factor=False)
        else:
            scaled_x = input_images

        scaled_image_shape = scaled_x.shape[2:]

        # Calculate DVFs through network
        scaled_disp = self.disp_gen_model(scaled_x)  # (1,group*dim,d,h,w)

        # Convert DVFs to new shape, (group, dim=3, d, h, w). The 2nd dim of DVF follows (d,h,w) order
        # Fill in row-by-row (first the dim dimension, then the group dimension)
        scaled_disp = torch.squeeze(scaled_disp, 0).reshape(self.group, self.dim, *scaled_image_shape)

        # Up-sample DVFs
        if self.scale < 1:
            # disp = F.interpolate(scaled_disp, size=ori_input_shape, mode='bilinear' if self.dim == 2 else 'trilinear',
            #                      align_corners=True)
            disp = ResizeTransform(size=ori_input_shape, factor=1.0 / self.scale, dim=self.dim)(scaled_disp)
        else:
            disp = scaled_disp

        # Apply DVFs to original inputs.
        # img (group,1,d,h,w), disp (group,3,d,h,w) --> (group,1,d,h,w)
        deformed_image = self.spatial_transform(torch.transpose(input_images, 0, 1), disp)
        template_image = torch.mean(deformed_image, 0, keepdim=True)  # (1, 1, d, h, w)

        res = {'warped_input': deformed_image}
        if dvf:
            res['dvf'] = disp
        if template:
            res['template'] = template_image
        return res
