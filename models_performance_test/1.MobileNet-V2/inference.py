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
test_dataset = datasets.MNIST(root=dataset_path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# 加载 MobileNetV2 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 10)  # MNIST 有 10 个类别
model.load_state_dict(torch.load('mobilenet_v2_mnist.pth',weights_only=True))  # 加载之前保存的模型
model = model.to(device)

# 使用测试数据进行推理
model.eval()
test_iter = iter(test_loader)  # 迭代器，用于取出测试数据集中的一批数据
images, labels = next(test_iter)  # 取出一批数据
images, labels = images.to(device), labels.to(device)

# 推理
with torch.no_grad():  # 不需要计算梯度
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# 打印实际标签和预测结果
print("Actual labels: ", labels[:10].cpu().numpy())
print("Predic labels: ", predicted[:10].cpu().numpy())
