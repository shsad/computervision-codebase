import torch
from torch.utils.data import Dataset
from torch.optim import Optimizer
from PIL import Image
import natsort
import os
import skimage.io as io
import pandas as pd

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class CustomDataSet(Dataset):
    '''
    Dataset class for NIPS2017 images.
    '''
    def __init__(self, main_dir, transform, labels):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.labels = labels

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.labels['ImageId'][idx]) + '.png'
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return (tensor_image, self.labels['TrueLabel'][idx] - 1)


class NesterovMomentumSGD:
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p.data) for p in self.params if p is not None]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # 1. Compute the update
            self.velocity[i] = self.momentum * self.velocity[i] + param.grad

            # 2. Calculate the gradient
            corrected_gradient = param.grad + self.momentum * self.velocity[i]

            # 3. Update the parameters
            param.data.add_(corrected_gradient, alpha=-self.lr)