import argparse
import os

from utils import *
import numpy as np
import torchvision.transforms as transforms
from torchvision import utils
from torch import Tensor
from PIL import Image
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.SUFT_Fre13_3 import *
from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from data.middlebury_dataloader import Middlebury_dataset

import cv2
import os

import torch.nn as nn
import torch.nn.functional as F
import torch

net = net = SUFT_network(num_feats=40, kernel_size=3, scale=8)
net.load_state_dict(torch.load("/opt/data/private/SGNet/SGNet_X8.pth", map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])
dataset = NYU_v2_datset(root_dir='/opt/data/share/120106010699/nyu_data', scale=8, transform=data_transform, train=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
data_num = len(dataloader)

rmse = np.zeros(449)
mad = 0.0
test_minmax = np.load('/opt/data/share/120106010699/nyu_data/test_minmax.npy')
with torch.no_grad():
    net.eval()
    for idx, data in enumerate(dataloader):
        guidance, lr, gt = data['guidance'].cuda(), data['lr'].cuda(), data['gt'].cuda()
        out, out_grad = net((guidance, lr))
        minmax = test_minmax[:, idx]
        minmax = torch.from_numpy(minmax).cuda()
        rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)
        print(rmse[idx])
    print(rmse.mean())



