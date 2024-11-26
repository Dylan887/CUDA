import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 将图像缩小为1000x1000像素
    transforms.ToTensor(),  # 转换为 Tensor
])

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

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)
model.load_state_dict(torch.load('srcnn_set5.pth',weights_only=True))
model.eval()

# 加载测试图像并进行推理
test_dir = '../data/Set5_LR'  # 低分辨率图像文件夹
test_images = [f for f in os.listdir(test_dir) if f.endswith('.png') or f.endswith('.jpg')]

correct = 0
total = 0

for img_name in test_images:
    lr_img_path = os.path.join(test_dir, img_name)
    low_res_image = Image.open(lr_img_path).convert('L')  # 转换为灰度图像
    input_tensor = transform(low_res_image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 计算准确率（这里假设有高分辨率的 ground truth）
    hr_img_path = os.path.join('../data/Set5', img_name)
    high_res_image = transform(Image.open(hr_img_path).convert('L')).unsqueeze(0).to(device)

    # 简单计算准确率（以 MSE 作为衡量标准，越小越好）
    mse = nn.functional.mse_loss(output, high_res_image).item()
    if mse < 0.01:  # 假设一个阈值来判断是否正确预测
        correct += 1
    total += 1

    # 显示预测结果
    print(f'Processed image: {img_name}, MSE: {mse:.4f}')

accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
print('Finished Inference')
