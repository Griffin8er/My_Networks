import torch
import torch.utils.data as tud
from torch import nn, optim
from tqdm import tqdm
import torchvision.datasets as tvds
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

nc = 3
f_maps = 64
device='mps:0'
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

train_data = tvds.ImageFolder('cities/normal', transform=T.Compose([
                                            T.Resize((256, 512)),
                                            T.ToTensor(),
                                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))

masks = tvds.ImageFolder('cities/masks', transform=T.Compose([
                                            T.Resize((256, 512)),
                                            T.ToTensor(),
                                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))

dataset = tud.DataLoader(train_data, batch_size=20)
masked_data = tud.DataLoader(masks,batch_size=20)


class Dice(nn.Module):
    def __init__(self, smooth=1e-5):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_act):
        self.y_pred = y_pred
        self.y_act = y_act
        self.intersect = torch.sum(self.y_act*self.y_pred)
        self.union = torch.sum(self.y_act) + torch.sum(self.y_pred)
        return 1 - ((2.0 * self.intersect + self.smooth)/(self.union + self.smooth))

    def backward(self):
        return (2.0 * (self.union * self.y_act - self.intersect * (self.y_act + self.y_pred))
                + self.smooth)/torch.pow((self.union + self.smooth), 2)


class UNET_arch(nn.Module):
    def __init__(self):
        super(UNET_arch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, f_maps, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps, f_maps, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(f_maps, f_maps*2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps*2, f_maps*2, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps*2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f_maps*2, f_maps*4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps*4, f_maps*4, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps*4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(f_maps*4, f_maps*8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps*8, f_maps*8, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps*8)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(f_maps * 8, f_maps * 16, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps * 16, f_maps * 16, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps*16)
        )
        self.maxpool = nn.MaxPool2d((2, 2), 2)
        self.convT = nn.ConvTranspose2d(f_maps*16, f_maps*8, 2, 2, 0)
        self.convT2 = nn.ConvTranspose2d(f_maps*8, f_maps*4, 2, 2,0)
        self.convT3 = nn.ConvTranspose2d(f_maps*4, f_maps*2, 2, 2, 0)
        self.convT4 = nn.ConvTranspose2d(f_maps*2, f_maps, 2, 2, 0)
        self.conv1d = nn.Sequential(
            nn.Conv2d(f_maps*16, f_maps*8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps*8, f_maps*8, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps*8)
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(f_maps * 8, f_maps * 4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps * 4, f_maps * 4, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps*4)
        )
        self.conv3d = nn.Sequential(
            nn.Conv2d(f_maps * 4, f_maps*2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps*2, f_maps*2, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps*2)
        )
        self.conv4d = nn.Sequential(
            nn.Conv2d(f_maps * 2, f_maps, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(f_maps, f_maps, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(f_maps),
            nn.Conv2d(f_maps, nc, 3, 1, 1)
        )

    def forward(self, x):
        layer1 = self.conv(x)
        layer1_pool = self.maxpool(layer1)
        layer2 = self.conv2(layer1_pool)
        layer2_pool = self.maxpool(layer2)
        layer3 = self.conv3(layer2_pool)
        layer3_pool = self.maxpool(layer3)
        layer4 = self.conv4(layer3_pool)
        layer4_pool = self.maxpool(layer4)
        layer5 = self.conv5(layer4_pool)
        decode1 = self.convT(layer5)
        decode1_cat = torch.cat([layer4, decode1], dim=1)
        decode1_forward = self.conv1d(decode1_cat)
        decode2 = self.convT2(decode1_forward)
        decode2_cat = torch.cat([layer3, decode2], dim=1)
        decode2_forward = self.conv2d(decode2_cat)
        decode3 = self.convT3(decode2_forward)
        decode3_cat = torch.cat([layer2, decode3], dim=1)
        decode3_forward = self.conv3d(decode3_cat)
        decode4 = self.convT4(decode3_forward)
        decode4_cat = torch.cat([layer1, decode4], dim=1)
        output_1 = self.conv4d(decode4_cat)
        output = nn.functional.sigmoid(output_1)
        return output


gen = UNET_arch()
gen.to(device)

'''images, _ = next(iter(dataset))
images = images.to(device)
real_batch = gen(images)
real_batch = real_batch.to('cpu')
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(
np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=0, normalize=True).cpu(),
                                 (1, 2, 0)))
plt.show()'''

loss_fn = Dice()
optimizer = optim.Adam(gen.parameters(), lr=.0002, betas=(.5, .95))


for i in range(2):
    for j in range(5):
        images, _ = next(iter(dataset))
        targets, _ = next(iter(masked_data))

        optimizer.zero_grad()
        data = images.to(device)
        targets = targets.to(dtype=torch.float, device=device)

        prediction = gen(data)
        loss = loss_fn(prediction, targets)
        print(loss.item())
        loss.backward()

        optimizer.step()
        print(j)


images, _ = next(iter(dataset))
images = images.to(device)
real_batch = gen(images)
real_batch = real_batch.to('cpu')
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(
np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=0, normalize=True).cpu(),
                                 (1, 2, 0)))
plt.show()