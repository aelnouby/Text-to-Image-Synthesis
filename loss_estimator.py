import torch
from torch import  nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class generator_loss(torch.nn.Module):
    def __init__(self):
        super(generator_loss, self).__init__()
        self.estimator = nn.BCELoss()

    def forward(self, fake):
        batch_size = fake.size()[0]
        self.labels = Variable(torch.FloatTensor(batch_size).cuda().fill_(1))
        return self.estimator(fake, self.labels)

class discriminator_loss(torch.nn.Module):
    def __init__(self):
        super(discriminator_loss, self).__init__()
        self.estimator = nn.BCELoss()

    def forward(self, real, wrong, fake):
        batch_size = real.size()[0]
        self.real_labels = Variable(torch.FloatTensor(batch_size).cuda().fill_(1))
        self.fake_labels = Variable(torch.FloatTensor(batch_size).cuda().fill_(0))
        return self.estimator(real, self.real_labels) + 0.5 * (self.estimator(wrong, self.fake_labels) + self.estimator(fake, self.fake_labels))

