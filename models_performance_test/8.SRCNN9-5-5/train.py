import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 Tensor
])

# 修改 Set5 数据集类，添加 resize 步骤
class Set5Dataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(lr_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        lr_img_name = os.path.join(self.lr_dir, self.image_files[idx])
        hr_img_name = os.path.join(self.hr_dir, self.image_files[idx])
        low_res_image = Image.open(lr_img_name).convert('L')  # 转换为灰度图像
        high_res_image = Image.open(hr_img_name).convert('L')  # 转换为灰度图像

        # 调整图像大小为统一尺寸，例如 256x256
        low_res_image = low_res_image.resize((256, 256), Image.BICUBIC)
        high_res_image = high_res_image.resize((256, 256), Image.BICUBIC)

        if self.transform:
            high_res_image = self.transform(high_res_image)
            low_res_image = self.transform(low_res_image)
        return low_res_image, high_res_image  # 输入是低分辨率图像，目标是高分辨率图像


# 创建数据集和数据加载器
train_dataset = Set5Dataset(lr_dir='../data/Set5_LR', hr_dir='../data/Set5', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

# 定义 SRCNN 模型
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 加载 SRCNN 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 3
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader, 0):
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # 使用输入图像和输出图像之间的均方误差作为损失

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # 每 10 个 batch 打印一次 loss
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'srcnn_set5.pth')  # 保存模型
