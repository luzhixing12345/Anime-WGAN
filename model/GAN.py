import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.BaseModule import BasicModel

# model structure comes from https://github.com/Zeleni9/pytorch-wgan

class GAN(BasicModel):
    def __init__(self, cfg):
        
        super(GAN, self).__init__(cfg)
        # Generator architecture
        self.G = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.input_size),
            nn.LeakyReLU(0.2),
            nn.Tanh())

        # Discriminator architecture
        self.D = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

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

                # Flatten image
                images = images.view(self.batch_size, -1)
                
                # initialize random noise
                z = torch.rand((self.batch_size, 100))
                
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
                
                z = Variable(torch.randn(self.batch_size, 100).to(self.device))
                
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

                    z = Variable(torch.randn(self.batch_size, 100).to(self.device))

                    # log losses and save images
                    info = {
                        'd_loss': d_loss.data,
                        'g_loss': g_loss.data
                    }
                    self.logger.log_losses(info, generator_iter)

                    info = {
                        'real_images': self.reshape_images(images),
                        'generated_images': self.reshape_images(self.G(z))
                    }

                    self.logger.log_images(info, generator_iter)
                    self.save_model(epoch, generator_iter)
            epoch_end_time = time.time()
            print("Epoch time: %.2f" % (epoch_end_time - epoch_start_time))
            
        self.logger.save()

        end_time = time.time()
        print("Total time: %.2f" % (end_time - start_time))
        # Save the trained parameters
        self.save_model(epoch, generator_iter)
        