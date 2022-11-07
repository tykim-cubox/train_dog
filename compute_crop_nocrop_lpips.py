from torchvision.transforms.transforms import RandomCrop, RandomHorizontalFlip
from eval import load_params
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from torchvision import transforms
import os
import random
import argparse
from tqdm import tqdm

from models import Generator
from operation import load_params, InfiniteSamplerWrapper

noise_dim = 256
device = torch.device('cuda:%d'%(0))



transform_list = [
            transforms.Resize((int(256),int(256))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
trans = transforms.Compose(transform_list)

transform_aug = [
            transforms.Resize((int(256*1.2),int(256*1.2))),
            transforms.RandomCrop((int(256),int(256))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
trans = transforms.Compose(transform_list)
trans_aug = transforms.Compose(transform_aug)


data_root = '../../images/skulls'
dataset_1 = ImageFolder(root=data_root, transform=trans)
dataset_2 = ImageFolder(root=data_root, transform=trans_aug)

import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

all_dis = []
for i in tqdm(range(len(dataset_1))):
    real_iamge = dataset_1[i][0].unsqueeze(0).to(device)
    aug_iamge = dataset_2[i][0].unsqueeze(0).to(device)
    with torch.no_grad():
        dis = percept(aug_iamge, real_iamge).sum().cpu()
        all_dis.append(dis.view(1))

print(all_dis)
print(torch.cat(all_dis).mean())
