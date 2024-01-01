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

from models.SGNet import *
from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from data.middlebury_dataloader import Middlebury_dataset

import cv2
import os

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=16, help='scale factor')
parser.add_argument("--num_feats", type=int, default=40, help="channel number of the middle hidden layer")
parser.add_argument("--root_dir", type=str, default='/datapath/RGB-D-D', help="root dir of dataset")
parser.add_argument("--model_dir", type=str, default="/SGNet/ckpt/SGNet_X8_R.pth", help="path of model")
parser.add_argument("--results_dir", type=str, default='SGNet/results', help="root dir of results")

opt = parser.parse_args()

net = SGNet(num_feats=opt.num_feats, kernel_size=3, scale=opt.scale)
net.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

data_transform = transforms.Compose([transforms.ToTensor()])

dataset_name = opt.root_dir.split('/')[-1]
if dataset_name == 'nyu_data':
    dataset = NYU_v2_datset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform, train=False)
    test_minmax = np.load('%s/test_minmax.npy' % opt.root_dir)
    rmse = np.zeros(449)
elif dataset_name == 'RGB-D-D':
    dataset = RGBDD_Dataset(root_dir=opt.root_dir, scale=opt.scale, downsample='sync', train=False, transform=data_transform)
    rmse = np.zeros(405)
elif dataset_name == 'Middlebury':
    dataset = Middlebury_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform)
    rmse = np.zeros(30)
elif dataset_name == 'Lu':
    dataset = Middlebury_dataset(root_dir=opt.root_dir, scale=opt.scale, transform=data_transform)
    rmse = np.zeros(6)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
data_num = len(dataloader)


with torch.no_grad():
    net.eval()
    if dataset_name == 'nyu_data':
        for idx, data in enumerate(dataloader):
            guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
            out, out_grad = net((guidance, lr))
            minmax = test_minmax[:, idx]
            minmax = torch.from_numpy(minmax).cuda()
            rmse[idx] = calc_rmse(gt[0, 0], out[0, 0], minmax)
            
            path_output = '{}/output'.format(opt.results_dir)
            os.makedirs(path_output, exist_ok=True)
            path_save_pred = '{}/{:010d}.png'.format(path_output, idx)
            
            # Save results  (Save the output depth map)
            pred = out[0,0] * (minmax[0] - minmax[1]) + minmax[1]
            pred = pred * 1000.0
            pred = pred.cpu().detach().numpy()
            pred = pred.astype(np.uint16)
            pred = Image.fromarray(pred)
            pred.save(path_save_pred)
            
            # visualization  (Visual depth map)
            #pred = out[0, 0]
            #pred = pred.cpu().detach().numpy()
            #cv2.imwrite(path_save_pred, pred * 255.0)   
            
            print(rmse[idx])
        print(rmse.mean())
    elif dataset_name == 'RGB-D-D':
        for idx, data in enumerate(dataloader):
            guidance, lr, gt, maxx, minn = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device), data[
                'max'].to(device), data['min'].to(device)
            out, out_grad = net((guidance, lr))
            minmax = [maxx, minn]
            rmse[idx] = rgbdd_calc_rmse(gt[0, 0], out[0, 0], minmax)
            
            path_output = '{}/output'.format(opt.results_dir)
            os.makedirs(path_output, exist_ok=True)
            path_save_pred = '{}/{:010d}.png'.format(path_output, idx)
            
            # Save results  (Save the output depth map)
            pred = out[0, 0] * (maxx - minn) + minn
            pred = pred.cpu().detach().numpy()
            pred = pred.astype(np.uint16)
            pred = Image.fromarray(pred)
            pred.save(path_save_pred)
            
            # visualization  (Visual depth map)
            #pred = out[0, 0]
            #pred = pred.cpu().detach().numpy()
            #cv2.imwrite(path_save_pred, pred * 255.0)   
            print(rmse[idx])
        print(rmse.mean())
    elif (dataset_name == 'Middlebury') or (dataset_name == 'Lu'):
        for idx, data in enumerate(dataloader):
            guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
            out, out_grad = net((guidance, lr))
            rmse[idx] = midd_calc_rmse(gt[0, 0], out[0, 0])
            
            path_output = '{}/output'.format(opt.results_dir)
            os.makedirs(path_output, exist_ok=True)
            path_save_pred = '{}/{:010d}.png'.format(path_output, idx)
            
            # Save results  (Save the output depth map)
            pred = out[0,0] * 255.0
            pred = pred.cpu().detach().numpy()
            pred = pred.astype(np.uint16)
            pred = Image.fromarray(pred)
            pred.save(path_save_pred)

            # visualization  (Visual depth map)
            #pred = out[0, 0]
            #pred = pred.cpu().detach().numpy()
            #cv2.imwrite(path_save_pred, pred * 255.0)   

            print(rmse[idx])
        print(rmse.mean())
