import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from models import Generator


# if __name__ == '__main__':
#   noise_dim = 256
#   im_size = 256

#   net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=im_size)#, big=args.big )
#   net_ig.to('cuda')

#   ckpt = f"/home/aiteam/tykim/DE-GAN/trial_dog/models/80000.pth"
#   checkpoint = torch.load(ckpt, map_location=lambda a,b: a)

#   checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
#   net_ig.load_state_dict(checkpoint['g'])

#   noise = torch.randn(16, noise_dim).to('cuda')
#   net_ig(noise)
#   torch.onnx.export(net_ig, args=(noise),
#                     f='./generator.onnx', verbose=True, input_names=['noise'], output_names=['images'], opset_version=15, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)#torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH) 

# del checkpoint


# dist = 'eval_%d'%(epoch)
# dist = os.path.join(dist, 'img')
# os.makedirs(dist, exist_ok=True)

# with torch.no_grad():
#   for i in tqdm(range(args.n_sample//args.batch)):
#     noise = torch.randn(args.batch, noise_dim).to(device)
#     g_imgs = net_ig(noise)[0]
#     g_imgs = F.interpolate(g_imgs, 512)
#     for j, g_img in enumerate( g_imgs ):
#         vutils.save_image(g_img.add(1).mul(0.5), 
#             os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))



## Infernecne ##
import onnxruntime

onnx_path = '/home/aiteam/tykim/DE-GAN/trial_dog/generator.onnx'

ort_session = onnxruntime.InferenceSession(onnx_path, providers=[
        # 'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
)
# input_tensor = torch.randn((1, 256))
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
# ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)


# providers = [
#     ('CUDAExecutionProvider', {
#         'device_id': 0,
#         'arena_extend_strategy': 'kNextPowerOfTwo',
#         'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
#         'cudnn_conv_algo_search': 'EXHAUSTIVE',
#         'do_copy_in_default_stream': True,
#     }),
#     'CPUExecutionProvider',
# ]

# session = ort.InferenceSession(model_path, providers=providers)


# print(int(3/2))
# print(int(67/2))
# print(3//2)
# print(67//2)

