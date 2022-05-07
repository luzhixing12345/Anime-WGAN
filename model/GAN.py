import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.logger import logger

# model structure comes from https://github.com/Zeleni9/pytorch-wgan

class GAN(nn.Module):
    def __init__(self, cfg):
        
        super(GAN, self).__init__()
        # Generator architecture
        
        self.cfg = cfg
        self.height = cfg.IMAGE.HEIGHT
        self.width = cfg.IMAGE.WIDTH
        self.channels = cfg.IMAGE.CHANNEL
        self.output_size = self.height * self.width * self.channels
        self.epochs = cfg.SOLVER.EPOCHS
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        self.device = cfg.MODEL.DEVICE
        
        self.G = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.output_size),
            nn.LeakyReLU(0.2),
            nn.Tanh())

        # Discriminator architecture
        self.D = nn.Sequential(
            nn.Linear(self.output_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

        # Binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

        self.model_checkpoint_dir = cfg.MODEL.CHECKPOINT_DIR

    def train(self, train_loader):
        self.t_begin = time.time()
        generator_iter = 0

        for epoch in range(self.epochs+1):
            self.logger = logger(self.cfg.OUTPUT_DIR, epoch)
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


                if ((i + 1) % 1000) == 0:
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

        self.logger.save()

        self.t_end = time.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # Save the trained parameters
        self.save_model(epoch, generator_iter)
        
    def reshape_images(self,data):
        
        number_of_images = self.cfg.IMAGE.NUMBER
        
        data = data.cpu().detach().numpy()[:number_of_images]
        images = []
        for sample in data:
            images.append(sample.reshape(self.channels, self.height, self.width))
        return images

    def save_model(self,epoch,iteration):
        torch.save(self.G.state_dict(), os.path.join(self.model_checkpoint_dir, 'G_epoch_{}_iter_{}.pth'.format(epoch,iteration)))
        torch.save(self.D.state_dict(), os.path.join(self.model_checkpoint_dir, 'D_epoch_{}_iter_{}.pth'.format(epoch,iteration)))
        print('Models save to {}'.format(self.model_checkpoint_dir))

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))