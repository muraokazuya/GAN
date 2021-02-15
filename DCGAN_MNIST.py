import torch
from torch import optim
from torch import nn
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    
def train(netD, netG, batch_size, zsize, epochs, trainloader):
    losses_netD = []
    losses_netG = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netD = netD.to(device)
    netG = netG.to(device)
    one_labels = torch.ones(batch_size).reshape(batch_size, 1).to(device)
    zero_labels = torch.zeros(batch_size).reshape(batch_size, 1).to(device)
    criterion = nn.BCELoss().to(device)

    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=[0.5, 0.999])
    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=[0.5, 0.999])

    for epoch in range(1, epochs+1):
        running_loss_netD = 0.0
        running_loss_netG = 0.0
        for count, (real_imgs, _) in enumerate(trainloader, 1):
            netD.zero_grad()

            # 識別器の学習
            real_imgs = real_imgs.to(device)
            # データローダーから読み込んだデータを識別器に入力し、損失を計算
            output_from_real = netD(real_imgs).reshape(batch_size, -1)
            loss_from_real = criterion(output_from_real, one_labels)
            loss_from_real.backward()

            # 生成器から得たデータを、識別器に入力し、損失を計算
            z = torch.randn(batch_size, zsize, 1, 1).to(device)
            fake_imgs = netG(z).to(device)
            output_from_fake = netD(fake_imgs.detach()).reshape(batch_size, -1)
            loss_from_fake = criterion(output_from_fake, zero_labels)
            loss_from_fake.backward()

            # それらをまとめたものが最終的な損失
            loss_netD = loss_from_real + loss_from_fake
            optD.step()
            running_loss_netD += loss_netD

            # 生成器の学習
            netG.zero_grad()
            z = torch.randn(batch_size, zsize, 1, 1).to(device)
            fake_imgs = netG(z).to(device)
            output_from_fake = netD(fake_imgs).reshape(batch_size, -1)
            loss_netG = criterion(output_from_fake, one_labels)
            loss_netG.backward()
            optG.step()
            running_loss_netG += loss_netG

        running_loss_netD /= count
        running_loss_netG /= count
        print(f'epoch: {epoch}, netD loss: {running_loss_netD}, netG loss: {running_loss_netG}')
        losses_netD.append(running_loss_netD)
        losses_netG.append(running_loss_netG)
        if epoch % 10 == 0:
            z = torch.randn(batch_size, zsize, 1, 1).to(device)
            generated_imgs = netG(z).cpu()
            imshow(generated_imgs[0:8].reshape(8, 1, 28, 28))
    return losses_netD, losses_netG

class MakeFrom(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.model = s
    def forward(self, x):
        return self.model(x)

###Transform input range to -1~+1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

###training data
batch_size = 100
trainset = MNIST('.', train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

###feature map size
feature_maps = 16
###noize size
zsize = 100

discriminator = nn.Sequential(
    nn.Conv2d(1, 16, 5, 2, bias=False),
    nn.LeakyReLU(0.2),
    nn.Conv2d(16, 32, 5, 2, bias=False),
    nn.LeakyReLU(0.2),
    nn.Conv2d(32, 64, 3, bias=False),
    nn.LeakyReLU(0.2),
    nn.Conv2d(64, 1, 2, bias=False),
    nn.Sigmoid()
)

generator = nn.Sequential(
    nn.ConvTranspose2d(zsize, feature_maps * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(feature_maps * 8),
    nn.ReLU(),
    nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(feature_maps * 4),
    nn.ReLU(),
    nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(feature_maps * 2),
    nn.ReLU(),
    nn.ConvTranspose2d(feature_maps * 2, 1, 2, 2, 2, bias=False),
    nn.Tanh()
)

netD = MakeFrom(discriminator)
netG = MakeFrom(generator)

EPOCHS = 80
losses_netD, losses_netG = train(netD, netG, batch_size, zsize, EPOCHS, trainloader)