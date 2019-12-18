
import torch
import torch.nn as nn
from torch.nn import utils
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                        nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2,padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),

                        nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2,padding=1, bias=False),
                        nn.Tanh()
                        )


    def forward(self, input):

        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(utils.spectral_norm(nn.Conv2d(3, 128, 4, stride=2, padding=1, bias=False)), # batch_size x 64 x 16 x 16
                            nn.LeakyReLU(0.2, inplace=True),

                            utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)), # batch_size x 128 x 8 x 8
                            nn.BatchNorm2d(256),
                            nn.LeakyReLU(0.2, inplace=True),

                            utils.spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)), # batch_size x 256 x 4 x 4
                            nn.BatchNorm2d(512),
                            nn.LeakyReLU(0.2, inplace=True),
                            
                            utils.spectral_norm(nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False)), # batch_size x 1 x 1 x 1
                            nn.Sigmoid()
                            )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)