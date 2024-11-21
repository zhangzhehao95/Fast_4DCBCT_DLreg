import SimpleITK as sitk
import numpy as np


class StopCriteriaConvergeStd(object):
    """
        Analyze the standard deviation of the latest query_len loss
    """
    def __init__(self, stop_std=0.0007, query_len=100, num_min_iter=200):
        self.query_len = query_len  # How many latest numbers will be analysed
        self.stop_std = stop_std
        self.loss_list = []
        self.loss_min = 2.
        self.num_min_iter = num_min_iter

    def add(self, loss):
        self.loss_list.append(loss)
        best_update = False
        if loss < self.loss_min:
            self.loss_min = loss
            best_update = True
        return best_update

    def stop(self):
        # Return True if the stop criteria are met
        query_list = self.loss_list[-self.query_len:]
        query_std = np.std(query_list)
        # If the iteration number exceed the predefined number, and
        # the latest numbers don't change too much, and
        # current loss is larger than previous minimum, and
        # current loss is smaller than previous minimum plus stop_std/3, and
        # , then stop.
        if len(self.loss_list) > self.num_min_iter and query_std < self.stop_std and \
                self.loss_min < self.loss_list[-1] < self.loss_min + self.stop_std / 3.:
            return True
        else:
            return False


class StopCriteriaConvergeLoss(object):
    """
        Analyze whether the loss converged to previous average / median
    """
    def __init__(self, difference=0.0001, query_len=10, num_min_iter=10, compare='median'):
        self.query_len = query_len  # How many latest numbers will be analysed
        self.difference = difference
        self.loss_list = []
        self.loss_min = 2.
        self.num_min_iter = num_min_iter
        self.compare = compare

    def add(self, loss):
        self.loss_list.append(loss)
        best_update = False
        if loss < self.loss_min:
            self.loss_min = loss
            best_update = True
        return best_update

    def stop(self):
        # Return True if the stop criteria are met
        query_list = self.loss_list[-self.query_len:]
        if self.compare == 'median':
            query_measure = np.median(query_list)
        else:
            query_measure = np.mean(query_list)
        # If the iteration number exceed the predefined number,
        # the difference between  current loss and previous average/median is smaller than threshold
        # , then stop.
        if len(self.loss_list) > self.num_min_iter and np.abs(self.loss_list[-1] - query_measure) < self.difference:
            return True
        else:
            return False


class StopCriteriaImprove(object):
    """
        Analyze whether the loss gets smaller in the latest query_len iterations
    """
    def __init__(self, min_improve=0, query_len=7, num_min_iter=10):
        self.patience = query_len    # How many latest numbers will be analysed
        self.min_improve = min_improve
        self.loss_list = []
        self.loss_min = 2.
        self.num_min_iter = num_min_iter
        self.cur_iter = 0

    def add(self, loss):
        self.cur_iter += 1
        best_update = False
        if loss < self.loss_min - self.min_improve:
            self.loss_min = loss
            self.loss_list = [loss]
            best_update = True
        else:
            self.loss_list.append(loss)
        return best_update

    def stop(self):
        # Return True if the stop criteria are met.
        # If the iteration number exceed the predefined number, and
        # loss doesn't have improvement at least min_improve in query_len iterations, then stop.
        if self.cur_iter >= self.num_min_iter and len(self.loss_list) > self.patience:
            return True
        else:
            return False


def save_as_itk(data, file_name, spacing=(2, 2, 2), direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), isVector=False):
    image = sitk.GetImageFromArray(data, isVector=isVector)
    image_size = image.GetSize()
    origin = [-(image_size[i] - 1)/2*spacing[i] for i in range(len(image_size))]
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    sitk.WriteImage(image, file_name)


# Crop out a smaller 3D volume from the center of data
def crop3D_mid(data, size):
    [l, m, n] = data.shape
    [out_l, out_m, out_n] = size
    start_l = (l - out_l) // 2
    start_m = (m - out_m) // 2
    start_n = (n - out_n) // 2
    res = data[start_l:start_l+out_l, start_m:start_m+out_m, start_n:start_n+out_n]
    return res


# Crop out a smaller 3D volume from the left-upper corner of data
def crop3D_luCorner(data, size):
    [out_l, out_m, out_n] = size
    res = data[:out_l, :out_m, :out_n]
    return res
