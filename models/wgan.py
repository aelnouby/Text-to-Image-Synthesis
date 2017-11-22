import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import Concat_embed, minibatch_discriminator
import pdb

class generator(nn.Module):
	def __init__(self):
		super(generator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.noise_dim = 100
		self.ngf = 64

		# based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
		self.netG = nn.Sequential(
			nn.ConvTranspose2d(self.noise_dim, self.ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False),
			nn.Tanh()
			 # state size. (num_channels) x 64 x 64
			)

	def forward(self, z):

		output = self.netG(z)
		return output

class discriminator(nn.Module):
	def __init__(self, improved = False):
		super(discriminator, self).__init__()
		self.image_size = 64
		self.num_channels = 3
		self.ndf = 64

		if improved:
			self.netD_1 = nn.Sequential(
				# input is (nc) x 64 x 64
				nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf) x 32 x 32
				nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*2) x 16 x 16
				nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*4) x 8 x 8
				nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*8) x 4 x 4
			)
		else:
			self.netD_1 = nn.Sequential(
				# input is (nc) x 64 x 64
				nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf) x 32 x 32
				nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
				nn.BatchNorm2d(self.ndf * 2),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*2) x 16 x 16
				nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
				nn.BatchNorm2d(self.ndf * 4),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*4) x 8 x 8
				nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
				nn.BatchNorm2d(self.ndf * 8),
				nn.LeakyReLU(0.2, inplace=True),
				# state size. (ndf*8) x 4 x 4
			)


		self.netD_2 = nn.Sequential(
			# nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
			)

	def forward(self, inp):
		x_intermediate = self.netD_1(inp)
		x = self.netD_2(x_intermediate)
		x = x.mean(0)

		return x.view(1) , x_intermediate