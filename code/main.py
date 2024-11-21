import sys
import random
import numpy as np

from argparse import ArgumentParser
from configuration.configuration import Configuration
from utils.logger import Logger
from model.interface import *


def main(config_file='configure_default.py'):
    parser = ArgumentParser(description='Registration Network')
    parser.add_argument('-c', '--config_file', type=str, default=config_file, help='Path to config file', required=True)
    parser.add_argument('-d', '--data_path', type=str, help='Path to dataset', required=True)
    parser.add_argument('-r', '--result_path', type=str, help='Path to result (output)', required=True)
    args = parser.parse_args()

    # Load configuration file
    configuration = Configuration(args.config_file, args.data_path, args.result_path)
    cf = configuration.load()

    # Modify the print function to save the printed info
    if cf.train_model:
        sys.stdout = Logger(cf.log_file)
        print('\n > Configuration file path: ' + cf.config_file)

    # Set up random seed
    seed = cf.manual_seed
    if seed is None:
        seed = random.randint(1, 10000)
    print('\n > Utilized random seed: ' + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)     # For multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n > Utilized device: ' + str(device))

    # Train the model, with validation
    if cf.train_model:
        print('\n > Start training process...')
        if cf.reg_model == 'Group':
            groupwise_train(device, cf)
        elif cf.reg_model == 'Pair':
            # TO BE DONE
            pass

    # Test the pre-trained model
    if cf.test_model:
        print('\n > Start test process...')
        if cf.reg_model == 'Group':
            groupwise_test(device, cf)


if __name__ == '__main__':
    main()


