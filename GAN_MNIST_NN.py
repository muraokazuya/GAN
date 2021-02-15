import torch
from torch import optim
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()
    
class Discriminator(torch.nn.Module):
    def _init_weights(self):
        for weight in self.parameters():
            torch.nn.init.normal_(weight, 0.0, 0.02)
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 384, bias=False)
        self.fc2 = torch.nn.Linear(384, 128, bias=False)
        self.fc3 = torch.nn.Linear(128, 32, bias=False)
        self.fc4 = torch.nn.Linear(32, 1, bias=False)
        self.relu = torch.nn.LeakyReLU()
        self._init_weights()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # 0 to 1(0: fake, 1: true)
        return x

class Generator(torch.nn.Module):
    def _init_weights(self):
        for weight in self.parameters():
            torch.nn.init.normal_(weight, 0.0, 0.02)
    def __init__(self, zsize, in_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(zsize, 256, bias=False)
        self.fc2 = torch.nn.Linear(256, 512, bias=False)
        self.fc3 = torch.nn.Linear(512, 1024, bias=False)
        self.fc4 = torch.nn.Linear(1024, in_features, bias=False)
        self.relu = torch.nn.ReLU()
        self._init_weights()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # -1 to 1
        return x

###input size
in_features = 1 * 28 * 28
###noize size
zsize = 100

###Transform input range to -1~+1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
###Training data
batch_size = 100

trainset = MNIST('.', train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

###Loss and epoch
losses_netD = []
losses_netG = []
EPOCHS = 50

###For debug, to see the discriminator works
"""
netD = Discriminator(in_features)
iterator = iter(trainloader)
img, _ = next(iterator)
D_out = netD(img.reshape(batch_size, -1))
print(D_out[0:5])
"""

###CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using: {device}')

###Discriminator and generator
netD = Discriminator(in_features).to(device)
netG = Generator(zsize, in_features).to(device)
criterion = torch.nn.BCELoss().to(device)

###Labels for training discriminator and generator
one_labels = torch.ones(batch_size).to(device)
zero_labels = torch.zeros(batch_size).to(device)

###Optimizer
optimizer_netD = optim.Adam(netD.parameters(), lr=0.0002, betas=[0.5, 0.999])
optimizer_netG = optim.Adam(netG.parameters(), lr=0.0002, betas=[0.5, 0.999])


for epoch in range(1, EPOCHS+1):
    running_loss_netD = 0.0
    running_loss_netG = 0.0
    for count, (real_imgs, _) in enumerate(trainloader, 1):
        netD.zero_grad()

        # 識別器の学習
        real_imgs = real_imgs.to(device)

        # データローダーからデータを読み込み、識別器に入力し、損失を計算
        output_real_imgs = netD(real_imgs.reshape(batch_size, -1))
        output_real_imgs = output_real_imgs.reshape(batch_size)
        loss_real_imgs = criterion(output_real_imgs, one_labels)
        loss_real_imgs.backward()

        # 生成器から得たデータを、識別器に入力し、損失を計算
        z = torch.randn(batch_size, zsize).to(device)
        fake_imgs = netG(z)
        output_fake_imgs = netD(fake_imgs.detach()).reshape(batch_size)
        loss_fake_imgs = criterion(output_fake_imgs, zero_labels)
        loss_fake_imgs.backward()

        # それらをまとめたものが最終的な損失
        loss_netD = loss_real_imgs + loss_fake_imgs
        optimizer_netD.step()
        running_loss_netD += loss_netD

        # 生成器の学習
        netG.zero_grad()
        z = torch.randn(batch_size, zsize).to(device)
        fake_imgs = netG(z)
        output_fake_imgs = netD(fake_imgs).reshape(batch_size)
        loss_netG = criterion(output_fake_imgs, one_labels)
        loss_netG.backward()
        optimizer_netG.step()
        running_loss_netG += loss_netG

    running_loss_netD /= count
    running_loss_netG /= count
    print(f'epoch: {epoch}, netD loss: {running_loss_netD}, netG loss: {running_loss_netG}')
    losses_netD.append(running_loss_netD.cpu())
    losses_netG.append(running_loss_netG.cpu())
    if epoch % 10 == 0:
        z = torch.randn(batch_size, zsize).to(device)
        generated_imgs = netG(z).cpu()
        imshow(generated_imgs[0:8].reshape(8, 1, 28, 28))