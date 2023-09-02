
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
import random

#import cv2
#from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import os



class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		self.main = nn.Sequential(
			# in: latent_size x 1 x 1
			nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Dropout(p=0.3),

			# out: 512 x 4 x 4

			nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.Dropout(p=0.3),
			# out: 256 x 8 x 8

			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.Dropout(p=0.3),
			# out: 128 x 16 x 16

			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Dropout(p=0.3),
			# out: 64 x 32 x 32

			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.Dropout(p=0.3),
			# out: 32 x 64 x 64

			nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
			nn.Tanh()
			# out: 3 x 128 x 128
		)

	def forward(self, input):
		x = self.main(input)
		return x



class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.main = nn.Sequential(
			# in: 3 x 128 x 128
	#		GaussianNoise(),
			nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),
			#         nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			# out: 32 x 64 x 64

			GaussianNoise(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			# out: 64 x 32 x 32

			GaussianNoise(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			# out: `128 x 16 x 16

			GaussianNoise(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			# out: 256 x 8 x 8

			GaussianNoise(),
			nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),
			# out: 512 x 4 x 4

			GaussianNoise(),
			nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
			# out: 1 x 1 x 1

			nn.Flatten()
		)

	def forward(self, input):
		return self.main(input)


class GaussianNoise(nn.Module):

    def __init__(self, std=0.05, decay_rate=0):
      super().__init__()
      self.std = std
      self.decay_rate = decay_rate

    def decay_step(self):
      return max(self.std - self.decay_rate, 0)

    def forward(self, x):
      if self.training:
        std = self.decay_step()
        return x + torch.empty_like(x).normal_(std=std)
      else:
        return x


class ImageBuffer:

    def __init__(self, bufsize):
        self.bufsize = bufsize
        self.n_images = 0
        self.buff = []

    def extract(self, images):

        return_images = []

        for image in images:
            if self.n_images < self.bufsize:
                return_images.append(image)
                self.buff.append(image)
                self.n_images += 1
            else:
                if random.uniform(0, 1) >= 0.5:
                    return_images.append(image)
                else:
                    idx = random.randint(0, self.bufsize-1)
                    img_return = self.buff[idx].clone()
                    return_images.append(img_return)
                    self.buff[idx] = image
        return torch.stack(return_images, dim=0)
