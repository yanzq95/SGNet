import numpy as np
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from data.common import arugment

def get_patch(img, lr, gt, scale, patch_size=16):
    th, tw = img.shape[:2]  ## HR image

    tp = round(patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))
    lr_tx = tx // scale
    lr_ty = ty // scale
    lr_tp = tp // scale

    return img[ty:ty + tp, tx:tx + tp], lr[lr_ty:lr_ty + lr_tp, lr_tx:lr_tx + lr_tp], gt[ty:ty + tp, tx:tx + tp]

class RGBDD_Dataset(Dataset):
    """RGB-D-D Dataset."""

    def __init__(self, root_dir, scale=4, downsample='real', train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            downsample (str): kernel type of downsample, real mean use real LR and HR data
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        types = ['models', 'plants', 'portraits']


        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.downsample = downsample
        self.train = train
        
        if train:
            if self.downsample == 'real':
                self.GTs = []
                self.LRs = []
                self.RGBs = []
                for type in types:
                    list_dir = os.listdir('%s/%s/%s_train'% (root_dir, type, type))
                    for n in list_dir:
                        self.RGBs.append('%s/%s/%s_train/%s/%s_RGB.jpg' % (root_dir, type, type, n, n))
                        self.GTs.append('%s/%s/%s_train/%s/%s_HR_gt.png' % (root_dir, type, type, n, n))
                        self.LRs.append('%s/%s/%s_train/%s/%s_LR_fill_depth.png' % (root_dir, type, type, n, n))
            else:
                self.GTs = []
                self.RGBs = []
                for type in types:
                    list_dir = os.listdir('%s/%s/%s_train'% (root_dir, type, type))
                    for n in list_dir:
                        self.RGBs.append('%s/%s/%s_train/%s/%s_RGB.jpg' % (root_dir, type, type, n, n))
                        self.GTs.append('%s/%s/%s_train/%s/%s_HR_gt.png' % (root_dir, type, type, n, n))

        else:
            if self.downsample == 'real':
                self.GTs = []
                self.LRs = []
                self.RGBs = []
                for type in types:
                    list_dir = os.listdir('%s/%s/%s_test'% (root_dir, type, type))
                    for n in list_dir:
                        self.RGBs.append('%s/%s/%s_test/%s/%s_RGB.jpg' % (root_dir, type, type, n, n))
                        self.GTs.append('%s/%s/%s_test/%s/%s_HR_gt.png' % (root_dir, type, type, n, n))
                        self.LRs.append('%s/%s/%s_test/%s/%s_LR_fill_depth.png' % (root_dir, type, type, n, n))
            else:
                self.GTs = []
                self.RGBs = []
                for type in types:
                    list_dir = os.listdir('%s/%s/%s_test'% (root_dir, type, type))
                    for n in list_dir:
                        self.RGBs.append('%s/%s/%s_test/%s/%s_RGB.jpg' % (root_dir, type, type, n, n))
                        self.GTs.append('%s/%s/%s_test/%s/%s_HR_gt.png' % (root_dir, type, type, n, n))

    def __len__(self):
        return len(self.GTs)

    def __getitem__(self, idx):
        if self.downsample == 'real':
            image =  np.array(Image.open(self.RGBs[idx]).convert("RGB")).astype(np.float32)
            gt = np.array(Image.open(self.GTs[idx])).astype(np.float32)

            h, w = gt.shape
            s = self.scale
            lr = np.array(Image.open(self.LRs[idx]).resize((w//s,h//s),Image.BICUBIC)).astype(np.float32)

        else:
            image = Image.open(self.RGBs[idx]).convert("RGB")
            image = np.array(image).astype(np.float32)
            gt = Image.open(self.GTs[idx])
            w, h = gt.size
            s = self.scale
            lr = np.array(gt.resize((w//s,h//s),Image.BICUBIC)).astype(np.float32)
            gt = np.array(gt).astype(np.float32)

        # normalization
        maxx = np.max(lr)
        minn = np.min(lr)
        lr=(lr-minn)/(maxx-minn)
        
        image_max = np.max(image)
        image_min = np.min(image)
        image = (image-image_min)/(image_max-image_min)

        if self.train:
            max_out = np.max(gt)
            min_out = np.min(gt)
            gt=(gt-min_out)/(max_out-min_out)
            image, lr, gt = get_patch(img=image, lr=np.expand_dims(lr, 2), gt=np.expand_dims(gt, 2), scale=self.scale, patch_size=256)

        if self.transform:
            image = self.transform(image).float()
            gt = self.transform(gt).float()
            lr = self.transform(lr).float()
        sample = {'guidance': image, 'lr': lr, 'gt': gt, 'max': maxx, 'min': minn}
        
        return sample