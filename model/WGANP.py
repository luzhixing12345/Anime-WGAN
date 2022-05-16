import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from model.BaseModule import BasicGAN


class Generator(torch.nn.Module):
    def __init__(self, dimension = 768, input_size = 100):
        super().__init__()
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            
            # Use nn.Upsample instead of nn.ConvTranspose2d to avoid Checkerboard Artifacts
            # see more information about Checkboard Artifacts in https://distill.pub/2016/deconv-checkerboard/
            
            # input is Z which size is (batch size x C x 1 X 1),going into a convolution
            # by default (32, 100, 1, 1)
            # project and reshape
            nn.ConvTranspose2d(in_channels=input_size, out_channels=dimension, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=dimension),
            nn.ReLU(True),

            # CONV1
            # State (32,1024,4,4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=dimension, out_channels=dimension//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=dimension//2),
            nn.ReLU(True),

            # CONV2
            # State (32,512,8,8)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=dimension//2, out_channels=dimension//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=dimension//4),
            nn.ReLU(True),
            
            # CONV3
            # State (32,256,16,16)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=dimension//4, out_channels=dimension//8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=dimension//8),
            nn.ReLU(True),
            
            # CONV4
            # State (32,128,32,32)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=dimension//8, out_channels=3, kernel_size=3, stride=1, padding=1),
            )
            # State (32,3,64,64)

        # output activation function is Tanh
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels = 3, dimension = 256):
        super().__init__()
        # Input_dim = channels (CxHxW)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # State (batch size x C x H x W)
            # default (32, 3, 64, 64)
            nn.Conv2d(in_channels=channels, out_channels=dimension, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dimension, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (32,256,32,32)
            nn.Conv2d(in_channels=dimension, out_channels=dimension*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dimension*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (32,512,16,16)
            nn.Conv2d(in_channels=dimension*2, out_channels=dimension*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dimension*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (32,1024,8,8)
            nn.Conv2d(in_channels=dimension*4, out_channels=dimension*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(dimension*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State (32,2048,4,4)
            )
        
        
            # outptut of main module --> State (8192x 4 x 4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=dimension*8, out_channels=1, kernel_size=4, stride=1, padding=0))
            # remove sigmoid function
            # output size (1 x (H/16-3) x (W/16-3))
            # default (32, 1, 1, 1)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)



class WGANP(BasicGAN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.G = Generator()
        self.D = Discriminator()

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS)

        self.generator_iters = cfg.MODEL.WGAN.GENERATOR_ITERS
        self.critic_iter = cfg.MODEL.WGAN.CRITIC_ITERS
        self.lambda_term = cfg.MODEL.WGAN.LAMBDA

    def train(self, train_loader):
    
        start_time = time.time()
        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = (one * -1).to(self.device)

        for g_iter in range(self.generator_iters):

            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.randn(self.batch_size, self.G_input_size,1,1).to(self.device)
                images = images.to(self.device)
                images = Variable(images)
                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = torch.randn(self.batch_size, self.G_input_size,1,1).to(self.device)
                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                #d_loss.backward()
                #Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
            
                # print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake.data}, loss_real: {d_loss_real.data}')



            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            # Train generator
            # Compute loss with fake images
            z = torch.randn(self.batch_size, self.G_input_size,1,1).to(self.device)
            
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()

            if ((g_iter + 1) % self.checkpoint_freq) == 0:
                self.logger.log(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss.data.item()}')
                z = torch.randn(self.batch_size, self.G_input_size,1,1).to(self.device)

                # log losses and save images
                info = {
                    'd_loss': d_loss.data,
                    'g_loss': g_loss.data
                }
                self.logger.log_losses(info, g_iter+1)

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

                self.logger.log_images(info, g_iter+1)
                self.save_model(g_iter)
                self.record_fake_images()
                    
        end_time = time.time()
        self.logger.log("Total time: %.2f" % (end_time - start_time))
        # Save the trained parameters
        self.save_model(g_iter)
        self.logger.save()

    def get_infinite_batches(self, data_loader):
        while True:
            for _, images in enumerate(data_loader):
                yield images
    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).to(self.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = interpolated.to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty