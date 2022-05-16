import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from model.BaseModule import BasicGAN
from .generator import Generator
from .discriminator import Discriminator

# model structure is in discriminator.py and generator.py
# have not finished yet

class WGAN256(BasicGAN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.G = Generator(self.channels,self.G_dimension,self.G_input_size)
        self.D = Discriminator(self.channels,self.D_dimension)

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
                print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss.data.item()}')
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
        print("Total time: %.2f" % (end_time - start_time))
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