import random
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)


device="mps:0"

dataset = dset.ImageFolder(root='my_face',
                           transform=T.Compose([
                               T.Resize((64, 64)),
                               T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

data_set = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

real_batch = next(iter(data_set))
plt.figure(figsize=(32,32))
plt.axis('off')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64],
                                         padding=2, normalize=True).cpu(), (1, 2, 0)))


def weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.stack = nn.Sequential(
            nn.ConvTranspose2d(100, 64*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.stack(input)


gen = Generator(1).to(device)
gen.apply(weights)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64*2),
            nn.Conv2d(64*2, 64*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64*4),
            nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64*8),
            nn.Conv2d(64*8, 1, 4, 1, 0, bias=False),
            nn.Softmax()
        )

    def forward(self, input):
        return self.model(input)


disc = Discriminator(1).to(device)
disc.apply(weights)


loss_fn = nn.BCELoss()

fixed_noise = torch.randn(64, 100, 1, 1, device=device)

real_label = 1.
fake_label = 0.

opt_gen = torch.optim.Adam(gen.parameters(), lr=.0002, betas=(.5, .9))
opt_disc = torch.optim.Adam(disc.parameters(), lr=.0002, betas=(.5, .9))

img_list = []
G_losses = []
D_losses = []
iters = 0

epochs = 150

print("Starting Training Loop...")
for epoch in range(epochs):
    for i, data in enumerate(data_set, 0):
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float32)
        output = disc(real_cpu).view(-1)
        errD_real = loss_fn(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 100, 1, 1)
        fake = gen(noise)
        label.fill_(fake_label)
        output = disc(fake.detach()).view(-1)
        errD_fake = loss_fn(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        disc.step()
        gen.zero_grad()
        label.fill_(real_label)
        output = disc(fake)
        errG = loss_fn(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        opt_gen.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(data_set),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(data_set) - 1)):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
