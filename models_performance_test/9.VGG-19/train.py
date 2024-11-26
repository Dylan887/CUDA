import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为适合 VGG19 的输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差进行标准化
])

# 自定义数据集加载器
class Set5Dataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.image_names = [f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        lr_img_name = os.path.join(self.lr_dir, self.image_names[idx])
        hr_img_name = os.path.join(self.hr_dir, self.image_names[idx])
        lr_image = Image.open(lr_img_name).convert('RGB')
        hr_image = Image.open(hr_img_name).convert('RGB')
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        return lr_image, hr_image

# 加载数据集（使用 Set5 数据集的低分辨率和高清图像）
train_dataset = Set5Dataset(lr_dir='../data/Set5_LR', hr_dir='../data/Set5', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

# 加载 VGG19 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19(weights=None)  # 不使用预训练权重
model.classifier[6] = nn.Linear(4096, 3 * 224 * 224)  # 输出大小与高清图像一致
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 3
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view_as(labels)  # 调整输出形状与标签一致
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个 batch 打印一次 loss
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'vgg19_set5.pth')  # 保存模型
