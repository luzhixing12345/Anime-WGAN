import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.BaseModule import BasicGAN

# model structure comes from https://github.com/Zeleni9/pytorch-wgan

class Generator(torch.nn.Module):
    def __init__(self, channels, dimension = 1024, input_size = 100):
        super().__init__()
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            
            # input is Z which size is (batch size x C x 1 X 1),going into a convolution
            # by default (32, 100, 1, 1)
            # project and reshape
            nn.ConvTranspose2d(in_channels=input_size, out_channels=dimension, kernel_size=4, stride=1, padding=0),
            nn.ReLU(True),

            # CONV1
            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=dimension, out_channels=dimension//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),

            # CONV2
            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=dimension//2, out_channels=dimension//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # CONV3
            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=dimension//4, out_channels=dimension//8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            
            # CONV4
            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=dimension//8, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (batch size x C x 64 x 64)
            # default (32, 3, 64, 64)

        # output activation function is Tanh
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels,dimension):
        super().__init__()
        # Input_dim = channels (CxHxW)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # State (CxHxW)
            # default (32, 3, 64, 64)
            nn.Conv2d(in_channels=channels, out_channels=dimension, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dimension),
            nn.ReLU(),

            # State (256x H/2 x W/2)
            nn.Conv2d(in_channels=dimension, out_channels=dimension*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dimension*2),
            nn.ReLU(),

            # State (512x H/4 x W/4)
            nn.Conv2d(in_channels=dimension*2, out_channels=dimension*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dimension*4),
            nn.ReLU(),
            
            # State (1024x H/8 x W/8)
            nn.Conv2d(in_channels=dimension*4, out_channels=dimension*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dimension*8),
            nn.ReLU())
            # outptut of main module --> State (2048x H/16 x W/16)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=dimension*8, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())
        
            # output size (1 x (H/16-3) x (W/16-3))
            # default (32, 1, 1, 1)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)



class GAN(BasicGAN):
    def __init__(self, cfg):
        
        super(GAN, self).__init__(cfg)
        # Generator and Discriminator architecture
        self.G = Generator(self.channels,self.G_dimension,self.G_input_size)
        self.D = Discriminator(self.channels,self.D_dimension)

        # Binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)


    def train(self, train_loader):
        
        start_time = time.time()
        generator_iter = 0
        
        for epoch in range(self.epochs+1):
            epoch_start_time = time.time()
            for i, images in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break
                
                # initialize random noise
                z = torch.rand((self.batch_size, self.G_input_size,1,1))
                
                real_labels = Variable(torch.ones(self.batch_size)).to(self.device)
                fake_labels = Variable(torch.zeros(self.batch_size)).to(self.device)
                images, z = Variable(images.to(self.device)), Variable(z.to(self.device))

                # Train discriminator
                # compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # [Training discriminator = Maximizing discriminator being correct]
                outputs = self.D(images)
                d_loss_real = self.loss(outputs.flatten(), real_labels)
                # real_score = outputs

                # Compute BCELoss using fake images
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                # fake_score = outputs

                # Optimizie discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                
                z = Variable(torch.randn(self.batch_size, self.G_input_size).to(self.device))
                
                fake_images = self.G(z)
                outputs = self.D(fake_images)

                # We train G to maximize log(D(G(z))[maximize likelihood of discriminator being wrong] instead of
                # minimizing log(1-D(G(z)))[minizing likelihood of discriminator being correct]
                # From paper  [https://arxiv.org/pdf/1406.2661.pdf]
                g_loss = self.loss(outputs.flatten(), real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1


                if ((i + 1) % self.checkpoint_freq) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data, g_loss.data))

                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)

                    # log losses and save images
                    info = {
                        'd_loss': d_loss.data,
                        'g_loss': g_loss.data
                    }
                    self.logger.log_losses(info, generator_iter)

                    with torch.no_grad():
                        fake_images = self.G(z)[:self.number_of_images]
                        real_images = images[:self.number_of_images]
                        # discriminate real images and fake images
                        fake_labels = self.D(fake_images).flatten()
                        real_labels = self.D(images).flatten()

                    info = {
                        'real_images': real_images.cpu().detach().numpy(),
                        'fake_images': fake_images.cpu().detach().numpy(),
                        'real_labels': real_labels.cpu().detach().numpy(),
                        'fake_labels': fake_labels.cpu().detach().numpy()
                    }

                    self.logger.log_images(info, epoch)
                    self.save_model(epoch)
                    
                    
            end_epoch_time = time.time()
            print("Epoch time: %.2f" % (end_epoch_time - epoch_start_time))
            self.record_fake_images()
        end_time = time.time()
        print("Total time: %.2f" % (end_time - start_time))
        # Save the trained parameters
        self.save_model(epoch)