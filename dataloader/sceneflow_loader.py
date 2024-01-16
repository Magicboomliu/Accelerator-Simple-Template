from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os

from dataloader.utils import read_text_lines
from dataloader.file_io import read_disp,read_img
# from utils.utils import read_text_lines
# from utils.file_io import read_disp,read_img
from skimage import io, transform
import numpy as np

class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 train_datalist,
                 test_datalist,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.img_size=(540, 960)
        self.scale_size =(576,960)
        

        sceneflow_finalpass_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'train_all': 'filenames/KITTI_2012_train_all.txt',
            'val': 'filenames/KITTI_2012_val.txt',
            'test': 'filenames/KITTI_2012_test.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': 'filenames/KITTI_mix.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_mix': kitti_mix_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]
            
            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None

            if load_pseudo_gt and sample['disp'] is not None:
                # KITTI 2015
                if 'disp_occ_0' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ_0',
                                                                     'disp_occ_0_pseudo_gt')
                # KITTI 2012
                elif 'disp_occ' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ',
                                                                     'disp_occ_pseudo_gt')
                else:
                    raise NotImplementedError
            else:
                sample['pseudo_disp'] = None

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['img_right'] = read_img(sample_path['right'])
    
    
        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['gt_disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]

        if self.mode=='test' or self.mode=='val':
            # img_left = transform.resize(sample['img_left'], [576,960], preserve_range=True)
            # img_right = transform.resize(sample['img_right'], [576,960], preserve_range=True)
            img_left = sample['img_left']
            img_right = sample['img_right']
            
            img_left = img_left.astype(np.float32)
            img_right = img_right.astype(np.float32)
            
            sample['img_left'] = img_left
            sample['img_right'] = img_right

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size