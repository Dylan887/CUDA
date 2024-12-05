import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import random

# 自定义数据集
class VOCDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.mask_dir = os.path.join(root_dir, "SegmentationClass")
        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

# SPADE 模块
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = nn.functional.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta

# SPADE Generator
class SPADEGenerator(nn.Module):
    def __init__(self, label_nc, ngf=64):
        super(SPADEGenerator, self).__init__()
        self.fc = nn.Linear(256, 16 * ngf * 4 * 4)
        self.head = SPADE(16 * ngf, label_nc)
        self.conv_img = nn.Conv2d(ngf, 3, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, segmap, z=None):
        if z is None:
            z = torch.randn(segmap.size(0), 256, device=segmap.device)

        x = self.fc(z).view(-1, 16 * 64, 4, 4)
        x = self.head(x, segmap)
        x = self.up(x)
        x = self.conv_img(x)
        return torch.tanh(x)

# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data/VOC2007"
batch_size = 8
epochs = 1  # 仅进行性能测试，训练 1 个 epoch

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# 加载数据
dataset = VOCDataset(root_dir, transform=transform, target_transform=target_transform)
subset_size = 100  # 使用部分数据进行测试
dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# 模型和损失函数
model = SPADEGenerator(label_nc=21).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

# 训练循环
for epoch in range(epochs):
    for i, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(masks)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

print("Training completed!")
torch.save(model.state_dict(), "spade_generator.pth")
