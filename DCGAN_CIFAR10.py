import torch
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 100
trainset = CIFAR10('.', train=True, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()

class MakeFrom(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.model = s
    def forward(self, x):
        return self.model(x)


def train(netD, netG, batch_size, zsize, epochs, trainloader):
    losses_netD = []
    losses_netG = []
    out_D_real = []
    out_D_fake = []
    out_G = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netD = netD.to(device)
    netG = netG.to(device)
    one_labels = torch.ones(batch_size).reshape(batch_size, 1).to(device)
    zero_labels = torch.zeros(batch_size).reshape(batch_size, 1).to(device)
    criterion = nn.BCELoss().to(device)#BCEloss outputs -100 for log0, so output range is 0~100.

    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=[0.5, 0.999])
    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=[0.5, 0.999])

    fixed_noise = torch.randn(8, zsize, 1, 1).to(device)

    for epoch in range(1, epochs+1):
        running_loss_netD = 0.0
        running_loss_netG = 0.0
        for count, (real_imgs, _) in enumerate(trainloader, 1):
            netD.zero_grad()

            # 識別器の学習
            real_imgs = real_imgs.to(device)
            # データローダーから読み込んだデータを識別器に入力し、損失を計算
            output_real = netD(real_imgs).reshape(batch_size, -1)
            loss_real = criterion(output_real, one_labels)
            loss_real.backward()

            # 生成器から得たデータを、識別器に入力し、損失を計算
            z = torch.randn(batch_size, zsize, 1, 1).to(device)
            fake_imgs = netG(z).to(device)
            output_fake1 = netD(fake_imgs.detach()).reshape(batch_size, -1)
            loss_fake1 = criterion(output_fake1, zero_labels)
            loss_fake1.backward()

            # それらをまとめたものが最終的な損失
            loss_netD = loss_real + loss_fake1
            optD.step()
            running_loss_netD += loss_netD  # 1バッチ分の損失の平均値を加算

            # 生成器の学習
            netG.zero_grad()
            z = torch.randn(batch_size, zsize, 1, 1).to(device)
            fake_imgs = netG(z).to(device)
            output_fake2 = netD(fake_imgs).reshape(batch_size, -1)
            loss_netG = criterion(output_fake2, one_labels)
            loss_netG.backward()
            optG.step()
            running_loss_netG += loss_netG # 1バッチ分の損失の平均値を加算

            # 最初のエポックだけ10、20、……、100バッチ終了時の学習状況を表示
            if epoch == 1:
                if count < 100 and count % 10 ==0:
                    stat1 = f'epoch: {epoch:02d}, batch: {count}\t'
                    stat2 = f'  lossD: {loss_netD:.4f}(real: {loss_real:.4f}, fake: {loss_fake1:.4f}),'
                    stat3 = f'lossG: {loss_netG:.4f},  D(x): {output_real.mean():.4f},'
                    stat4 = f'D(G(z)): {output_fake1.mean():.4f}, {output_fake2.mean():.4f}'
                    print(stat1, stat2, stat3, stat4)

            if count % 100 == 0:  # 1エポックの中で100回ごとに学習の状況を記録
                out_D_real.append(output_real.mean())
                out_D_fake.append(output_fake1.mean())
                out_G.append(output_fake2.mean())
                stat1 = f'epoch: {epoch:02d}, batch: {count}\t'
                stat2 = f'  lossD: {loss_netD:.4f}(real: {loss_real:.4f}, fake: {loss_fake1:.4f}),'
                stat3 = f'lossG: {loss_netG:.4f},  D(x): {output_real.mean():.4f},'
                stat4 = f'D(G(z)): {output_fake1.mean():.4f}, {output_fake2.mean():.4f}'
                print(stat1, stat2, stat3, stat4)

        running_loss_netD /= count  # 1エポック終了時にその間の損失の平均を求める
        running_loss_netG /= count
        losses_netD.append(running_loss_netD)
        losses_netG.append(running_loss_netG)
        print(f'epoch: {epoch}, running_loss_D: {running_loss_netD}, running_loss_G: {running_loss_netG}', '\n')
        if epoch % 5 == 0:
            generated_imgs = netG(fixed_noise).cpu()
            imshow(generated_imgs.reshape(8, 3, 32, 32))

    return (losses_netD, losses_netG), (out_D_real, out_D_fake, out_G), (netD, netG)

###Initialize weight
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

discriminator = nn.Sequential(
    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(256, 1, 4, 1, 0, bias=False),
    nn.Sigmoid()
)

feature_maps = 64
zsize = 100
generator = nn.Sequential(
    nn.ConvTranspose2d(zsize, feature_maps * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(feature_maps * 8),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(feature_maps * 4),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(feature_maps * 2),
    nn.ReLU(inplace=True),
    nn.ConvTranspose2d(feature_maps * 2, 3, 4, 2, 1, bias=False),
    nn.Tanh()
)

netD = MakeFrom(discriminator)
netG = MakeFrom(generator)
netD.apply(weights_init)
netG.apply(weights_init)

EPOCHS = 80
losses, outs, nets = train(netD, netG, batch_size, zsize, EPOCHS, trainloader)