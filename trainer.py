import numpy as np
import torch
from torch import nn
import yaml
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from gan_cls import generator, discriminator
import gan
from cub_dataset import CUBDataset
from loss_estimator import generator_loss, discriminator_loss
from visualize import VisdomPlotter
from utils import Utils
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
        self.beta1 = 0.5
        self.num_epochs = 600

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)

        self.viz = VisdomPlotter()
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def train_gan_conditional(self):
        criterion = nn.BCELoss()
        l1_loss = nn.L1Loss()
        lr = 0.0002
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.beta1, 0.999))
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(self.beta1, 0.999))

        for epoch in range(self.num_epochs):
            print("Epoch %d started."%(epoch))

            for sample in self.data_loader:
                self.discriminator.zero_grad()
                right_image = sample['right_image']
                right_emed = sample['right_emed']
                wrong_image = sample['wrong_image']

                right_image = Variable(right_image.float()).cuda()
                right_emed = Variable(right_emed.float()).cuda()

                real_labels = torch.ones(right_image.size(0))
                fake_labels = torch.zeros(right_image.size(0))

                # One sided label smoothing
                real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.3))

                real_labels = Variable(real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs = self.discriminator(right_image, right_emed)
                real_loss = criterion(outputs, real_labels)
                real_score = outputs

                noise = Variable(torch.randn(right_image.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_emed, noise)
                outputs = self.discriminator(fake_images, right_emed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss
                d_loss.backward()
                d_optimizer.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_image.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_emed, noise)
                outputs = self.discriminator(fake_images, right_emed)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                g_optimizer.step()

                self.viz.draw('generated images', fake_images.data.cpu().numpy()[:64] * 128 + 128)
                self.viz.draw('real images', right_image.data.cpu().numpy()[:64] * 128 + 128)

                print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
                epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(), fake_score.data.cpu().mean()))
                self.hist_D.append(d_loss.data.cpu().mean())
                self.hist_G.append(g_loss.data.cpu().mean())
                self.hist_Dx.append(real_score.data.cpu().mean())
                self.hist_DGx.append(fake_score.data.cpu().mean())

            self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
            self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
            self.viz.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).mean())
            self.viz.plot('D(G(X))', 'train', epoch, np.array(self.hist_DGx).mean())
            self.hist_D = []
            self.hist_G = []
            self.hist_Dx = []
            self.hist_DGx = []

            if (epoch+1)%50 == 0:
                torch.save(self.discriminator.state_dict(), 'models/' + str(epoch+1) + '_discrimnator.pth')
                torch.save(self.generator.state_dict(), 'models/' + str(epoch+1) + '_generator.pth')

    def train_gan_cls(self):
        criterion = nn.BCELoss()
        l1_loss = nn.L1Loss()
        lr = 0.0002
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(self.beta1, 0.999))
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(self.beta1, 0.999))

        for epoch in range(self.num_epochs):
            print("Epoch %d started." % (epoch))

            for sample in self.data_loader:
                self.discriminator.zero_grad()
                right_image = sample['right_image']
                right_emed = sample['right_emed']
                wrong_image = sample['wrong_image']

                right_image = Variable(right_image.float()).cuda()
                right_emed = Variable(right_emed.float()).cuda()
                wrong_image = Variable(wrong_image.float()).cuda()

                real_labels = torch.ones(right_image.size(0))
                fake_labels = torch.zeros(right_image.size(0))

                # One sided label smoothing
                real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.3))

                real_labels = Variable(real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs = self.discriminator(right_image, right_emed)
                real_loss = criterion(outputs, real_labels)
                real_score = outputs

                outputs = self.discriminator(wrong_image, right_emed)
                wrong_loss = criterion(outputs, fake_labels)
                wrong_score = outputs

                noise = Variable(torch.randn(right_image.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_emed, noise)
                outputs = self.discriminator(fake_images, right_emed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + 0.5 * (fake_loss + wrong_loss)
                d_loss.backward()
                d_optimizer.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_image.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_emed, noise)
                outputs = self.discriminator(fake_images, right_emed)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                g_optimizer.step()

                self.viz.draw('generated images', fake_images.data.cpu().numpy()[:64] * 128 + 128)
                self.viz.draw('real images', right_image.data.cpu().numpy()[:64] * 128 + 128)

                print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
                    epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
                    fake_score.data.cpu().mean()))
                self.hist_D.append(d_loss.data.cpu().mean())
                self.hist_G.append(g_loss.data.cpu().mean())
                self.hist_Dx.append(real_score.data.cpu().mean())
                self.hist_DGx.append(fake_score.data.cpu().mean())

            self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
            self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
            self.viz.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).mean())
            self.viz.plot('D(G(X))', 'train', epoch, np.array(self.hist_DGx).mean())
            self.hist_D = []
            self.hist_G = []
            self.hist_Dx = []
            self.hist_DGx = []

            if (epoch + 1) % 50 == 0:
                torch.save(self.discriminator.state_dict(), 'models/' + str(epoch + 1) + '_discrimnator.pth')
                torch.save(self.generator.state_dict(), 'models/' + str(epoch + 1) + '_generator.pth')

    def train_gan(self):
        generator = torch.nn.DataParallel(gan.generator().cuda())
        discriminator = torch.nn.DataParallel(gan.discriminator().cuda())
        criterion = nn.BCELoss()
        l1_loss = nn.L1Loss()
        lr = 0.0002
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(self.beta1, 0.999))
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(self.beta1, 0.999))

        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                images = sample['right_image']
                images = Variable(images, requires_grad=False).cuda()

                real_labels = torch.ones(images.size(0))
                fake_labels = torch.zeros(images.size(0))

                real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.3))

                real_labels = Variable(real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # Train the discriminator
                discriminator.zero_grad()
                outputs = discriminator(images)
                real_loss = criterion(outputs, real_labels)
                real_score = outputs

                noise = Variable(torch.randn(images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = generator(noise)
                outputs = discriminator(fake_images)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss
                d_loss.backward()
                d_optimizer.step()

                # Train the generator
                generator.zero_grad()
                noise = Variable(torch.randn(images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = generator(noise)
                outputs = discriminator(fake_images)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                g_optimizer.step()

                self.viz.draw('generated images', fake_images.data.cpu().numpy()[:64]*128+128)
                self.viz.draw('real images', images.data.cpu().numpy()[:64]*128+128)
         
                print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(),real_score.data.cpu().mean(), fake_score.data.cpu().mean()))
                self.hist_D.append(d_loss.data.cpu().mean())
                self.hist_G.append(g_loss.data.cpu().mean())

            self.viz.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
            self.viz.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
            self.hist_D = []
            self.hist_G = []









