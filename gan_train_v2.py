import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import glob
import os

# ------------------------------
# CONFIG
# ------------------------------
latent_dim = 100  # noise
image_size = 64
channels = 3  # rgb
batch_size = 32 # 128 in paper
epochs = 50 # is probably too much for MTG cards xd
lr = 0.0002 # and half for disc
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("generated", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# ------------------------------
# CUSTOM DATASET (no subfolders needed)
# ------------------------------
class CardDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = glob.glob(f"{folder}/all/*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*channels, [0.5]*channels)
])

dataset = CardDataset("./data/cards/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------------------
# MODELS
# ------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)
        return out.view(x.size(0), -1).mean(1)  # one scalar per image

# ------------------------------
# WEIGHT INITIALIZATION
# ------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# ------------------------------
# OPTIMIZERS & LOSS
# ------------------------------
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr/2, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# ------------------------------
# TRAINING LOOP
# ------------------------------
for epoch in range(epochs):
    for i, real in enumerate(dataloader):
        real = real.to(device)
        batch_size = real.size(0)
        label_real = torch.full((batch_size,), 0.9, device=device)
        label_fake = torch.zeros(batch_size, device=device)

        # --- Train Discriminator ---
        netD.zero_grad()
        output_real = netD(real)
        lossD_real = criterion(output_real, label_real)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake = netG(noise)
        output_fake = netD(fake.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # --- Train Generator ---
        netG.zero_grad()
        output = netD(fake)
        lossG = criterion(output, label_real)
        lossG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"[{epoch+1}/{epochs}][{i}/{len(dataloader)}] "
                  f"Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

    # --- Save samples and checkpoints ---
    with torch.no_grad():
        samples = netG(fixed_noise).detach().cpu()
        save_image(samples, f"generated/epoch_{epoch+1}.png", nrow=8, normalize=True)

    torch.save(netG.state_dict(), 'weights/netG.pth')
    torch.save(netD.state_dict(), 'weights/netD.pth')

print("Training complete! Check the 'generated/' folder for samples.")
