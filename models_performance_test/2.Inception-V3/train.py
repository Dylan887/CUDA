import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import inception_v3
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 将 28x28 调整为 299x299 适用于 InceptionV3
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道转换为三通道
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集标准化参数
])

# 加载 MNIST 数据集
train_dataset = torch.utils.data.Subset(datasets.MNIST(root='../data', train=True, download=False, transform=transform), range(1000))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

# 加载 InceptionV3 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_v3(weights=None)  # 不使用预训练权重
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 10)  # MNIST 有 10 个类别
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

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
        outputs, aux_outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4 * loss2  # InceptionV3 的损失函数结合了辅助分类器的输出

        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个 batch 打印一次 loss
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'inceptionv3_mnist.pth')  # 保存模型