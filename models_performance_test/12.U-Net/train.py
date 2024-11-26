import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 定义 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 定义编码器部分
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 定义中间层
        self.middle = self.conv_block(512, 1024)

        # 定义解码器部分
        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        # 定义输出层
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.downsample(enc1))
        enc3 = self.encoder3(self.downsample(enc2))
        enc4 = self.encoder4(self.downsample(enc3))

        # 中间层
        middle = self.middle(self.downsample(enc4))

        # 解码器部分
        dec4 = self.upsample(middle, enc4)
        dec4 = self.decoder4(torch.cat([dec4, enc4], dim=1))
        dec3 = self.upsample(dec4, enc3)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.upsample(dec3, enc2)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upsample(dec2, enc1)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))

        # 输出层
        return self.output_layer(dec1)

    def downsample(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)

    def upsample(self, x, target_feature_map):
        return nn.functional.interpolate(x, size=target_feature_map.shape[2:], mode='bilinear', align_corners=True)

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.image_names = [f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        lr_image_path = os.path.join(self.lr_dir, img_name)
        hr_image_path = os.path.join(self.hr_dir, img_name)
        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = Image.open(hr_image_path).convert('RGB')

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图像调整为适合 U-Net 的输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载数据集
lr_dir = '../data/Set5_LR'
hr_dir = '../data/Set5'
dataset = ImageDataset(lr_dir, hr_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

# 加载 U-Net 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 3
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, (lr_images, hr_images) in enumerate(train_loader, 0):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # 每 10 个 batch 打印一次 loss
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'unet_set5.pth')  # 保存模型
