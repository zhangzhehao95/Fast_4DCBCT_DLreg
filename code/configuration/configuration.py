from importlib.machinery import SourceFileLoader
import os
from os.path import join, exists
import datetime
import shutil


class Configuration:
    def __init__(self, config_file,
                 dataset_path,
                 result_path):

        self.config_file = config_file
        self.dataset_path = dataset_path
        self.result_path = result_path

    def load(self):
        # Load configuration file
        cf = SourceFileLoader('config', self.config_file).load_module()

        # Save extra parameters
        cf.config_file = self.config_file
        cf.dataset_path = self.dataset_path
        cf.result_path = self.result_path

        # Create output folder to save tensorboard, checkpoints, test results and log files
        cf.save_path = join(cf.result_path, cf.dataset_name, cf.experiment_name)
        if not exists(cf.save_path):
            os.makedirs(cf.save_path)

        # Define a log file and save print info into the log file
        cf.log_file = os.path.join(cf.save_path, "logfile.log")

        # Create folder to save weights
        cf.weights_save_path = join(cf.save_path, 'saved_model')
        if not exists(cf.weights_save_path):
            os.makedirs(cf.weights_save_path)

        # Weights to-be-used for testing
        if not hasattr(cf, "load_weights_file"):
            cf.load_weights_file = 'final_weights.pth'
        cf.load_weights_path = join(cf.weights_save_path, cf.load_weights_file)

        # Create folder to save test results
        if cf.test_model:
            cf.test_save_path = join(cf.save_path, 'test_results')
            if not exists(cf.test_save_path):
                os.makedirs(cf.test_save_path)

        # Create tensorboard folder
        cf.tb_dir = join(cf.save_path, 'tensorboard_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        # Default parameters
        if not hasattr(cf, "network_type"):
            cf.network_type = 'UNet'

        if not hasattr(cf, "load_pretrained"):
            cf.load_pretrained = False
        else:
            cf.pretained_weights_file = join(cf.weights_save_path, cf.load_weights_file)

        if not hasattr(cf, "simi_loss_type"):
            cf.simi_loss_type = 'NCC2'

        if not hasattr(cf, "smooth_loss_type"):
            cf.smooth_loss_type = 'dvf'

        if not hasattr(cf, "L2_penalty"):
            cf.L2_penalty = 0

        if not hasattr(cf, "downsampling"):
            cf.downsample = 'MaxPool'

        if not hasattr(cf, "upsampling"):
            cf.upsample = 'UpSample'

        if not hasattr(cf, "target_indexes"):
            cf.target_indexes = []

        if not hasattr(cf, "val_landmark_evl"):
            cf.val_landmark_evl = False

        if not hasattr(cf, "test_landmark_evl"):
            cf.test_landmark_evl = False

        if not hasattr(cf, "save_best_model"):
            cf.save_best_model = True   # weights will be saved to best_model.pth

        # Dataset paths
        cf.train_data_path = join(cf.dataset_path, cf.train_dir)
        cf.val_data_path = join(cf.dataset_path, cf.valid_dir)
        cf.test_data_path = join(cf.dataset_path, cf.test_dir)

        # Copy config file
        shutil.copyfile(cf.config_file, join(cf.save_path, "config.py"))

        return cf
