import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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
    transforms.Resize((224, 224)),  # 将图像调整为适合 ResNet-DPED 的输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差进行标准化
])

# 加载数据集
lr_dir = '../data/Set5_LR'
hr_dir = '../data/Set5'
dataset = ImageDataset(lr_dir, hr_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

# 加载 ResNet-DPED 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3 * 224 * 224)  # 输出与高分辨率图像大小匹配
model = model.to(device)

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
        outputs = outputs.view(-1, 3, 224, 224)  # 将输出重塑为图像大小
        loss = criterion(outputs, hr_images)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # 每 10 个 batch 打印一次 loss
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'resnet_dped_set5.pth')  # 保存模型
