import os
import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk


class PairDataset(Dataset):
    def __init__(self, data_path, phase_num=10):
        self.pair_dataset_names = []

        # Get all subdirectories of the given directory, data_path
        for sub_dir in os.listdir(data_path):
            subject_path = os.path.join(data_path, sub_dir)
            if os.path.isdir(subject_path):
                # Generate src and tar image pairs between any two phases, under ascending order
                for src_index in range(phase_num - 1):
                    src_path = os.path.join(subject_path, str(src_index) + '.mha')
                    for tar_index in range(src_index + 1, phase_num):
                        tar_path = os.path.join(subject_path, str(tar_index) + '.mha')
                        self.pair_dataset_names.append([src_path, tar_path])

        self.ids = len(self.pair_dataset_names)
        print(f'\n > Creating dataset with {self.ids } groups.')

    def __len__(self):
        return self.ids

    def __getitem__(self, idx):
        image_names = self.pair_dataset_names[idx]
        src_image_npy = sitk.GetArrayFromImage(sitk.ReadImage(image_names[0]))
        src_image = torch.from_numpy(src_image_npy[np.newaxis, ...]).type(torch.FloatTensor)
        tar_image_npy = sitk.GetArrayFromImage(sitk.ReadImage(image_names[1]))
        tar_image = torch.from_numpy(tar_image_npy[np.newaxis, ...]).type(torch.FloatTensor)

        # Convert ndarrays to Tensors
        return {'src_image': src_image, 'tar_image': tar_image}


class GroupDataset(Dataset):
    def __init__(self, data_path, phase_num=10):
        self.group_dataset_names = []

        # Get all subdirectories of the given data_path directory
        for sub_dir in os.listdir(data_path):
            subject_path = os.path.join(data_path, sub_dir)
            if os.path.isdir(subject_path):
                # Phase images: [0-9].mha under the same subject_path
                image_names = [os.path.join(subject_path, str(p) + '.mha') for p in range(phase_num)]
                self.group_dataset_names.append(image_names)

        self.ids = len(self.group_dataset_names)
        print(f'\n > Creating dataset with {self.ids } groups.')

    def __len__(self):
        return self.ids

    def __getitem__(self, idx):
        image_names = self.group_dataset_names[idx]     # Fetch one group of images
        image_npy = [sitk.GetArrayFromImage(sitk.ReadImage(item)).astype(np.float32) for item in image_names]
        # Convert ndarrays to Tensors and stack/concatenate a sequence of tensors along a new dimension.
        group_images = torch.stack([torch.from_numpy(image).type(torch.FloatTensor) for image in image_npy], 0)

        # Convert ndarrays to Tensors
        return {'images': group_images}
