import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # 将 28x28 调整为 224x224 适用于 MobileNetV2
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集标准化参数
])

# 加载 MNIST 数据集
dataset_path = '../data'
train_dataset = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)

# 只使用部分训练数据（例如 1000 个样本）
train_subset_size = 1000  # 只使用前 1000 个样本
train_subset = torch.utils.data.Subset(train_dataset, range(train_subset_size))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=2)
# 加载 MobileNetV2 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v2(weights=None)  # 不使用预训练权重
model.classifier[1] = nn.Linear(model.last_channel, 10)  # MNIST 有 10 个类别
model = model.to(device)
print(f"Model is using device: {device}")
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 3
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个 batch 打印一次 loss
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'mobilenet_v2_mnist.pth')  # 保存模型
