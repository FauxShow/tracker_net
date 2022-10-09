import os
from random import randint

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_image

class SiameseDataloader(Dataset):
    def __init__(self, data_root, train, input_w, input_h, device):
        super(SiameseDataloader, self).__init__()

        if train:
            self.data_root = os.path.join(data_root, 'train/')
        else:
            self.data_root = os.path.join(data_root, 'test/')

        self.input_w = input_w
        self.input_h = input_h
        self.device = device

        self.transforms = torch.nn.Sequential(
            T.Resize((input_w, input_h)),
            T.RandomHorizontalFlip(p=0.5)
        )

        self.all_class_dirs = [ os.path.join(self.data_root, c) for c in os.listdir(self.data_root) ]
        self.num_classes = len(self.all_class_dirs)


    def __len__(self):
        return self.num_classes ** 2


    def __getitem__(self, index):
       
        index = index % self.num_classes # otherwise the dataset appears tiny (= to num of class dirs)

        selected_class_idx = randint(0, self.num_classes-1)
        selected_class_dir = self.all_class_dirs[selected_class_idx]

        samples = os.listdir(selected_class_dir)
        sample_idx = randint(0, len(samples)-1)
        sample = os.path.join(selected_class_dir, samples[sample_idx])

        if index % 2 == 0:  # same class
            same_sample_idx = randint(0, len(samples)-1)
            while same_sample_idx == sample_idx: # ensure we dont pick the exact same sample
                same_sample_idx = randint(0, len(samples)-1)
    
            sample2 = os.path.join(selected_class_dir, samples[same_sample_idx])

            target = torch.tensor(1, dtype=torch.float) # 1 indicating these samples are of the same class
        else:   # different class
            diff_class_idx = randint(0, self.num_classes-1)
            while diff_class_idx == selected_class_idx:
                diff_class_idx = randint(0, self.num_classes-1)
            diff_class_dir = self.all_class_dirs[diff_class_idx]

            diff_samples = os.listdir(diff_class_dir)
            diff_sample_idx = randint(0, len(diff_samples)-1)
            sample2 = os.path.join(diff_class_dir, diff_samples[diff_sample_idx])

            target = torch.tensor(0, dtype=torch.float) # 0 indicating these samples are from different classes

        sample_im = read_image(sample).type(torch.float32)
        sample2_im = read_image(sample2).type(torch.float32)
        #sample_im = sample_im.to(self.device)
        #sample2_im = sample2_im.to(self.device)
        sample_im = self.transforms(sample_im)
        sample2_im = self.transforms(sample2_im)

        return sample_im, sample2_im, target



if __name__ == "__main__":
    import sys
    data_root = sys.argv[1]

    device = torch.device("cuda")
    dl = SiameseDataloader(data_root, 128, 128, device)

    pos_example = dl[0]
    neg_example = dl[1]

    # show same example
    cv2.imshow('anchor positive', pos_example[0])
    cv2.imshow('positive', pos_example[1])
    cv2.imshow('anchor negative', neg_example[0])
    cv2.imshow('negative', neg_example[1])

    cv2.waitKey(0)
