import torch
from torch.utils.data import DataLoader
import torchvision.datasets as tvd
import torchvision.utils as vutils
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim, save, max


device = 'mps:0' if torch.backends.mps.is_available() else 'default'

str = 'pets'

classes = ('Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
           'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua',
           'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
           'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger',
           'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian',
           'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier',
           'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier',
           'yorkshire_terrier')

dataset_tr = tvd.OxfordIIITPet(str, 'trainval', target_types="category", transform=T.Compose([
                                                T.Resize((64, 64)),
                                                T.ToTensor(),
                                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]), download=True)

dataset_t = tvd.OxfordIIITPet(str, 'test', target_types="category", transform=T.Compose([
                                                T.Resize((64, 64)),
                                                T.ToTensor(),
                                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]), download=True)

dataset_tr = DataLoader(dataset_tr, batch_size=128, shuffle=True)
dataset_t = DataLoader(dataset_t, batch_size=32, shuffle=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.MaxPool2d((1, 1)),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.MaxPool2d((1, 1)),
            nn.Dropout2d(),
            nn.Flatten(),
            nn.Linear(32768, 256),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(128, 37),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

model = Classifier()
model.apply(weights_init)
model = model.to(device)
model.eval()


loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), 0.001, betas=(.5, .91))

epochs = 40

for epoch in range(epochs):
    t_loss = 0
    for i, data in enumerate(dataset_tr, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.train()
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        t_loss+=loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
    print(f"Epoch {epoch} finished - loss: {t_loss/29}")


"save(model.state_dict(), 'state_dicts/state_dict_2')"

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
real_batch, _ = next(iter(dataset_t))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("test")
plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

plt.savefig('animal_classify')


real_batch = real_batch.to(device)
outputs = model(real_batch)

_, predicted = max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(32)))


plt.show()