import os
from random import randint

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image

class SiameseDataloader(Dataset):
    def __init__(self, data_root, mode, train, dataset_size, input_w, input_h, device):
        """ Mode is either siamese (for pairs) or triple (for (anchor, pos, neg)) """
        
        super(SiameseDataloader, self).__init__()

        self.mode = mode
        assert mode in ['siamese', 'triplet'], "Invalid mode [siamese, triplet]"

        if train:
            self.data_root = os.path.join(data_root, 'train/')
        else:
            self.data_root = os.path.join(data_root, 'test/')
        
        self.dataset_size = dataset_size
        self.input_w = input_w
        self.input_h = input_h
        self.device = device

        self.transforms = torch.nn.Sequential(
            T.Resize((input_w, input_h)),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        self.all_class_dirs = [ os.path.join(self.data_root, c) for c in os.listdir(self.data_root) ]
        self.num_classes = len(self.all_class_dirs)


    def __len__(self):
        """ Dataset length is arbitrary as the dataset pairs or triplets are selected at random 
        Even with small datasets, the total number of pairs or triplets is exponential in the number
        of actual images, so thi would be too large for a single epoch """
        return int(self.dataset_size)


    def __getitem__(self, index):
       
        index = index % self.num_classes # otherwise the dataset appears tiny (= to num of class dirs)

        selected_class_idx = randint(0, self.num_classes-1)
        selected_class_dir = self.all_class_dirs[selected_class_idx]

        samples = os.listdir(selected_class_dir)
        sample_idx = randint(0, len(samples)-1)
        sample = os.path.join(selected_class_dir, samples[sample_idx])

        # same class
        same_sample_idx = randint(0, len(samples)-1)
        while same_sample_idx == sample_idx: # ensure we dont pick the exact same sample
            same_sample_idx = randint(0, len(samples)-1)

        same_sample = os.path.join(selected_class_dir, samples[same_sample_idx])

        target = torch.tensor(1, dtype=torch.float) # 1 indicating these samples are of the same class
        
        # diff_class
        diff_class_idx = randint(0, self.num_classes-1)
        while diff_class_idx == selected_class_idx:
            diff_class_idx = randint(0, self.num_classes-1)
        diff_class_dir = self.all_class_dirs[diff_class_idx]

        diff_samples = os.listdir(diff_class_dir)
        diff_sample_idx = randint(0, len(diff_samples)-1)
        diff_sample = os.path.join(diff_class_dir, diff_samples[diff_sample_idx])

        if self.mode == 'siamese':

            target = torch.tensor(0, dtype=torch.float) # 0 indicating these samples are from different classes

            if index % 2 == 0:
                sample2 = same_sample
            else:
                sample2 = diff_sample

            sample_im = read_image(sample).type(torch.float32) / 255
            sample2_im = read_image(sample2).type(torch.float32) / 255
            sample_im = self.transforms(sample_im)
            sample2_im = self.transforms(sample2_im)

            return sample_im, sample2_im, target
        else:
            anchor_im = read_image(sample).type(torch.float32) / 255
            positive_im = read_image(same_sample).type(torch.float32) / 255
            negative_im = read_image(diff_sample).type(torch.float32) / 255
            anchor_im = self.transforms(anchor_im)
            positive_im = self.transforms(positive_im)
            negative_im = self.transforms(negative_im)
           
            class_labels = torch.tensor((selected_class_idx, selected_class_idx, diff_class_idx))

            return anchor_im, positive_im, negative_im, class_labels


if __name__ == "__main__":
    import sys
    import cv2
    import numpy as np

    data_root = sys.argv[1]

    device = torch.device("cuda")
    dl = SiameseDataloader(data_root, 'triplet', True, 128, 128, device)

    sample = dl[0]
    anchor = sample[0]
    pos_example = sample[1]
    neg_example = sample[2]
    
    # convert for display
    out_sample = anchor.numpy().astype(np.uint8)
    out_sample = np.transpose(out_sample, (1,2,0))
    out_pos = pos_example.numpy().astype(np.uint8)
    out_pos = np.transpose(out_pos, (1,2,0))
    out_neg = neg_example.numpy().astype(np.uint8)
    out_neg = np.transpose(out_neg, (1,2,0))

    # show same example
    cv2.imshow('anchor', out_sample)
    cv2.imshow('positive', out_pos)
    cv2.imshow('negative', out_neg)

    cv2.waitKey(0)
