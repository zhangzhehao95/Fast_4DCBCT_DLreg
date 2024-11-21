from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, StepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    """
    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler=None, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler  # Share the same optimizer with GradualWarmupScheduler
        self.finished = False   # Tag for after_scheduler initialization
        super(GradualWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # After warmup, follow the after_scheduler to adjust lr
        if self.last_epoch >= self.warmup_epoch:
            if self.after_scheduler:
                if not self.finished:   # Lr of after_scheduler starts from base_lr * multiplier
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.after_scheduler._last_lr = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * ((self.last_epoch + 1.) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1

        if self.last_epoch <= self.warmup_epoch:
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (self.last_epoch / self.warmup_epoch) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.warmup_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)  # after_scheduler.last_epoch begins to change after warmup
                else:
                    self.after_scheduler.step(epoch - self.warmup_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def LR_schedule(optimizer, scheduler_type='exp_decay', gamma=0.9, tol_epochs=100, arg=5):
    if scheduler_type == 'no_decay':
        lambda1 = lambda epoch: 1
        return LambdaLR(optimizer, lr_lambda=lambda1)

    elif scheduler_type == 'exp_decay':
        # lr_base * (gamma ** epoch)
        return ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_type == 'power_decay':
        # LambdaLR: return lr_base times a given function
        # lr_base * ((1 - epoch/tol_epochs) ** gamma)
        lambda1 = lambda epoch: (1 - epoch / tol_epochs) ** gamma
        return LambdaLR(optimizer, lr_lambda=lambda1)

    elif scheduler_type == 'step_decay':
        # Decays by gamma every step_size(arg) epochs
        # lr_base * (gamma ** floor(epoch/step_size))
        return StepLR(optimizer, step_size=arg, gamma=gamma)

    elif scheduler_type == 'boundary_decay':
        # start = lr_base, end = lr_base/arg, exponential decay
        decay_rate = np.power(1 / arg, 1 / (tol_epochs - 1))
        lambda1 = lambda epoch: decay_rate ** epoch
        return LambdaLR(optimizer, lr_lambda=lambda1)

    elif scheduler_type == 'plateau_decay':
        # Reduce learning rate when a metric has stopped improving.
        # scheduler.step(val_loss) to define which metric to monitor
        return ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5)

    elif scheduler_type == 'warmup_annealing_decay':
        scheduler_annealing = CosineAnnealingWarmRestarts(optimizer, T_0=tol_epochs-5, T_mult=1, eta_min=optimizer.param_groups[0]['lr'])
        # Init_lr increase to gamma*init_lr, then cosine annealing back to init_lr.
        scheduler = GradualWarmupScheduler(optimizer, multiplier=gamma, warmup_epoch=5, after_scheduler=scheduler_annealing)
        return scheduler

    else:
        raise ValueError('Unknown scheduler: ' + scheduler_type)
