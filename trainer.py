import numpy as np
import torch
import yaml
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gan_cls import generator, discriminator
from cub_dataset import CUBDataset
from loss_estimator import generator_loss, discriminator_loss
from visualize import VisdomPlotter
import pdb

class Trainer(object):
    def __init__(self):

        with open('config.yaml', 'r') as f:
            config = yaml.load(f)

        self.generator = torch.nn.DataParallel(generator().cuda())
        self.discriminator = torch.nn.DataParallel(discriminator().cuda())

        self.dataset = CUBDataset(config['birds_dataset_path'])

        self.noise_dim = 100
        self.batch_size = 64
        self.num_workers = 8
        self.lr = 0.0002
        self.num_epochs = 600

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)

        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.loss_G_estimator =  generator_loss()
        self.loss_D_estimator = discriminator_loss()
        self.viz = VisdomPlotter()


    def train(self):
        for epoch in range(self.num_epochs):
            print("Epoch %d started."%(epoch))

            for sample in self.data_loader:
                right_image = sample['right_image']
                right_emed = sample['right_emed']
                wrong_image = sample['wrong_image']


                right_image = Variable(right_image.float()).cuda()
                right_emed = Variable(right_emed.float()).cuda()
                wrong_image = Variable(wrong_image.float()).cuda()

                self.discriminator.zero_grad()

                logits_D_real = self.discriminator(right_image, right_emed)
                logits_D_wrong = self.discriminator(wrong_image, right_emed)

                z = Variable(torch.FloatTensor(right_emed.size()[0], self.noise_dim, 1, 1).normal_(0, 1).cuda())

                fake_image = self.generator(right_emed, z)
                self.viz.draw('generated images', fake_image.data.cpu().numpy())
                self.viz.draw('real images', right_image.data.cpu().numpy())

                logits_D_fake = self.discriminator(fake_image, right_emed)

                loss_D = self.loss_D_estimator(logits_D_real, logits_D_wrong, logits_D_fake)

                loss_D.backward()
                self.optim_D.step()

                self.generator.zero_grad()
                fake_image = self.generator(right_emed, z)
                logits_D_fake = self.discriminator(fake_image, right_emed)
                loss_G = self.loss_G_estimator(logits_D_fake)
                loss_G.backward()
                self.optim_G.step()


                print("Epoch: %d, discriminator loss: %f , Generator loss: %f" % (epoch, loss_D.data.cpu().mean(), loss_G.data.cpu().mean()))
                if epoch > 0 and epoch%50 == 0:
                    torch.save(self.generator.state_dict(), 'models/' + str(epoch) + '_generator.pth')
                    torch.save(self.discriminator.state_dict(), 'models/' + str(epoch) + '_discriminator.pth')







