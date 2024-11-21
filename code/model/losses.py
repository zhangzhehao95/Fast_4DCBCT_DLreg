import torch
import torch.nn as nn
import torch.nn.functional as F


##################
# Local/Over-window normalized cross-correlation (LNCC)
##################
class NCC_loss(nn.Module):
    """
    Calculate LNCC coefficient between two images.
        dim : Dimension of the input images.
        windows_size : Side length of the square window to calculate the local NCC.
    """

    def __init__(self, dim, windows_size=9):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.num_stab_const = 1e-5  # numerical stability constant

        self.windows_size = windows_size  # window size along one dimension
        self.pad = windows_size // 2
        self.window_volume = windows_size ** self.dim  # total voxel number in one window

        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def forward(self, I, J):
        """
        I/J:
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used.
        Returns:
            CC : Average local normalized cross-correlation coefficient.
        """
        try:
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)
        except:
            self.sum_filter = torch.ones([1, 1] + [self.windows_size, ] * self.dim, dtype=I.dtype, device=I.device)
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)

        J_sum = self.conv(J, self.sum_filter, padding=self.pad)  # (n, 1, h, w) or (n, 1, d, h, w)
        I2_sum = self.conv(I * I, self.sum_filter, padding=self.pad)
        J2_sum = self.conv(J * J, self.sum_filter, padding=self.pad)
        IJ_sum = self.conv(I * J, self.sum_filter, padding=self.pad)

        # Due to the precision issue, the I_var and I_var can be negative. Constrain to be larger than num_stab_const.
        cross = torch.clamp(IJ_sum - I_sum * J_sum / self.window_volume, min=self.num_stab_const)
        I_var = torch.clamp(I2_sum - I_sum ** 2 / self.window_volume, min=self.num_stab_const)
        J_var = torch.clamp(J2_sum - J_sum ** 2 / self.window_volume, min=self.num_stab_const)

        # The larger LNCC the better, use negated as loss
        cc = cross / torch.sqrt(I_var * J_var)
        return - torch.mean(cc)


class NCC2_loss(nn.Module):
    """
    Calculate the square of LNCC between two images.
        dim : Dimension of the input images.
        windows_size : Side length of the square window to calculate the local NCC.
    """

    def __init__(self, dim, windows_size=9):
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.num_stab_const = 1e-5  # numerical stability constant

        self.windows_size = windows_size
        self.pad = windows_size // 2
        self.window_volume = windows_size ** self.dim

        if self.dim == 2:
            self.conv = F.conv2d
        elif self.dim == 3:
            self.conv = F.conv3d

    def forward(self, I, J):
        """
        I/J:
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used.
        Returns:
            CC : Average local normalized cross-correlation coefficient.
        """
        try:
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)
        except:
            self.sum_filter = torch.ones([1, 1] + [self.windows_size, ] * self.dim, dtype=I.dtype, device=I.device)
            I_sum = self.conv(I, self.sum_filter, padding=self.pad)

        J_sum = self.conv(J, self.sum_filter, padding=self.pad)  # (n, 1, h, w) or (n, 1, d, h, w)
        I2_sum = self.conv(I * I, self.sum_filter, padding=self.pad)
        J2_sum = self.conv(J * J, self.sum_filter, padding=self.pad)
        IJ_sum = self.conv(I * J, self.sum_filter, padding=self.pad)

        cross = IJ_sum - I_sum * J_sum / self.window_volume
        I_var = I2_sum - I_sum * I_sum / self.window_volume
        J_var = J2_sum - J_sum * J_sum / self.window_volume

        # The larger the better, use negated as loss
        cc = (cross * cross) / (I_var * J_var + self.num_stab_const)
        return - torch.mean(cc)


##################
# MSE
##################
class MSE_loss(nn.Module):
    """
    Calculate mean squared error
    """

    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, X, Y):
        # The number of image in the first dimension can be different. Handel different input size by broadcasting
        """
        X/Y:
            Torch tensor of same shape. The number of image in the first dimension can be different, in which broadcasting will be used.
        """
        if X.shape[0] > Y.shape[0]:
            Y = Y.repeat([X.shape[0] // Y.shape[0]] + [1] * (len(Y.shape) - 1))
        elif X.shape[0] < Y.shape[0]:
            X = X.repeat([Y.shape[0] // X.shape[0]] + [1] * (len(X.shape) - 1))

        if not X.shape == Y.shape:
            raise ValueError(
                f"After broadcasting, input images should have the shape, but got {X.shape} and {Y.shape}.")

        return torch.mean((X - Y) ** 2)


##################
# Structural similarity -- SSIM
# https://github.com/VainF/pytorch-msssim
##################
def _gaussian_1d(size=11, sigma=1.5):
    """Create 1-D gaussian kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, window):
    """ Create gaussian window and blur input using multiple convolutions with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel, [C=1, 1, 1, 1, win_size]
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in window.shape[1:-1]]), window.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):  # [h,d,w]
        if s >= window.shape[-1]:
            # input tensor (batch, in_channels, iD, iH, iW)
            # kernel weights ((out_channels, in_channels/group, kD, kH , kW)
            out = conv(out, weight=window.transpose(2 + i, -1), stride=1, padding=0, groups=C)

    return out


def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    """ Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel, [C=1, 1, 1, 1, win_size]
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
    Returns:
        torch.Tensor: ssim results.
    """
    # X.shape: batch, channel, [depth,] height, width
    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # flatten from start_dim=2 to end_dim=-1, which will only keep [batch, channel, 1] dimension
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)

    return ssim_per_channel


def ssim(X, Y, data_range=1, size_average=True, win_size=11, win_sigma=1.5, win=None, K=(0.01, 0.03),
         nonnegative_ssim=False):
    """ interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,D,H,W)
        Y (torch.Tensor): a batch of images, (N,C,D,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if X.shape[0] > Y.shape[0]:
        Y = Y.repeat([X.shape[0] // Y.shape[0]] + [1] * (len(Y.shape) - 1))
    elif X.shape[0] < Y.shape[0]:
        X = X.repeat([Y.shape[0] // X.shape[0]] + [1] * (len(X.shape) - 1))

    if not X.shape == Y.shape:
        raise ValueError(f"After broadcasting, input images should have the shape, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _gaussian_1d(win_size, win_sigma)  # [1, 1, win_size]
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))  # [C=1, 1, 1, 1, win_size]

    ssim_per_channel = _ssim(X, Y, data_range=data_range, win=win, K=K)  # [batch, channel]
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


class SSIM_loss(nn.Module):
    def __init__(self, data_range=1, size_average=True, win_size=11, win_sigma=1.5, channel=1, spatial_dims=3,
                 K=(0.01, 0.03), nonnegative_ssim=False):
        """ class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (grey: 1)
            spatial_dims (int, optional): dimension number of image (2/3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """
        super(SSIM_loss, self).__init__()
        self.win_size = win_size
        self.win = _gaussian_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        ssim_index = ssim(X, Y, data_range=self.data_range, size_average=self.size_average, win=self.win, K=self.K,
                          nonnegative_ssim=self.nonnegative_ssim)
        # SSIM has range [-1,1], the larger the better. Use opposite as loss
        return - ssim_index


##################
# Calculate the spatial and temporal gradient of DVF
# https://github.com/DeepRegNet/DeepReg/blob/main/deepreg/loss/deform.py
# https://github.com/vincentme/GroupRegNet/blob/master/model/loss.py
##################
class SmoothRegularization(nn.Module):
    def __init__(self, s1w=1e-1, s2w=1e-1, t1w=1e-1, t2w=1e-1, mode='dvf', grad='forward'):
        """
        Use the central finite difference dx[i] = (x[i+1] - x[i-1]) / 2 or forward finite difference
        dx[i] = x[i+1] - x[i] to approximate gradient

        Parameters
            disp: (n=10, 3, d, h, w), displacement field
            image: (1, 1, d, h, w)
            mode: 'dvf' or 'dvf_image'
            grad: 'forward' or 'central'
        """
        super().__init__()
        self.s1w = s1w
        self.s2w = s2w
        self.t1w = t1w
        self.t2w = t2w
        self.mode = mode
        self.grad = grad

    def gradient_dx(self, disp):
        if self.grad == 'forward':
            return disp[:, :, 1:, :-1, :-1] - disp[:, :, :-1, :-1, :-1]
        else:
            return (disp[:, :, 2:, 1:-1, 1:-1] - disp[:, :, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, disp):
        if self.grad == 'forward':
            return disp[:, :, :-1, 1:, :-1] - disp[:, :, :-1, :-1, :-1]
        else:
            return (disp[:, :, 1:-1, 2:, 1:-1] - disp[:, :, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, disp):
        if self.grad == 'forward':
            return disp[:, :, :-1, :-1, 1:] - disp[:, :, :-1, :-1, :-1]
        else:
            return (disp[:, :, 1:-1, 1:-1, 2:] - disp[:, :, 1:-1, 1:-1, :-2]) / 2

    def gradient_dt(self, disp):
        if self.grad == 'forward':
            index_forward = list(range(1, disp.shape[0])) + [0, ]
            return disp[index_forward, :, :, :, :] - disp[:, :, :, :, :]
        else:
            time = disp.shape[0]
            index_forward = list(range(1, time)) + [0, ]
            index_backward = [-1, ] + list(range(time - 1))
            return (disp[index_forward, :, :, :, :] - disp[index_backward, :, :, :, :]) / 2

    def gradient_txyz(self, Txyz, fn):
        # This is the same as processing the xyz-dim separately and stacking them together
        return fn(Txyz)

    def forward(self, disp, image=None):
        if self.mode == 'dvf_image':
            # Adopted from GroupRegNet. Combine the gradients of DVF and image, which allows sharper edge displacement.
            # s1w as weight.
            # Use forward difference (Larger_Index - Smaller_Index).
            self.grad = 'forward'
            d_disp = torch.stack([self.gradient_txyz(disp, self.gradient_dz), self.gradient_txyz(disp, self.gradient_dy),
                                  self.gradient_txyz(disp, self.gradient_dx)], dim=1)

            # (1, 3, 1, d, h, w)
            d_image = torch.stack([self.gradient_txyz(image, self.gradient_dz), self.gradient_txyz(image, self.gradient_dy),
                                  self.gradient_txyz(image, self.gradient_dx)], dim=1)

            res = self.s1w * torch.mean(torch.sum(torch.abs(d_disp), dim=2, keepdims=True) * torch.exp(-torch.abs(d_image)))

        else:
            res = torch.tensor(0, dtype=disp.dtype, device=disp.device)
            # 1st order of spatial gradient
            dTdx = self.gradient_txyz(disp, self.gradient_dx)
            dTdy = self.gradient_txyz(disp, self.gradient_dy)
            dTdz = self.gradient_txyz(disp, self.gradient_dz)
            if self.s1w > 0:
                res += self.s1w * torch.mean(dTdx ** 2 + dTdy ** 2 + dTdz ** 2)
            # 2nd order of spatial gradient
            if self.s2w > 0:
                dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
                dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
                dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
                dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
                dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
                dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
                res += self.s2w * torch.mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2)
            # 1st order of temporal gradient
            if self.t1w > 0:
                dTdt = self.gradient_txyz(disp, self.gradient_dt)
                res += self.t1w * torch.mean(dTdt ** 2)
            # 2nd order of temporal gradient
            if self.t2w > 0:
                try:
                    dTdtt = self.gradient_txyz(dTdt, self.gradient_dt)
                except:
                    dTdt = self.gradient_txyz(disp, self.gradient_dt)
                    dTdtt = self.gradient_txyz(dTdt, self.gradient_dt)
                res += self.t2w * torch.mean(dTdtt ** 2)

        return res
