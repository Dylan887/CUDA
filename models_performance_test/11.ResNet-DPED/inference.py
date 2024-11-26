import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为适合 ResNet-DPED 的输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差进行标准化
])

# 加载 ResNet-DPED 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3 * 224 * 224)  # 输出与高分辨率图像大小匹配
model.load_state_dict(torch.load('resnet_dped_set5.pth',weights_only=True))  # 加载训练好的模型权重
model = model.to(device)
model.eval()

# 推理函数
def infer_image(lr_image_path, hr_image_path):
    lr_image = Image.open(lr_image_path).convert('RGB')
    hr_image = Image.open(hr_image_path).convert('RGB')
    input_tensor = transform(lr_image).unsqueeze(0).to(device)
    hr_tensor = transform(hr_image).unsqueeze(0).to(device)  # 确保高分辨率图像也在相同的设备上
    
    with torch.no_grad():
        output = model(input_tensor)
        output = output.view(1, 3, 224, 224).to(device)
    
    # 计算 MSE 作为准确度的指标之一
    mse_loss = nn.MSELoss()(output, hr_tensor)
    print(f"MSE Loss for {lr_image_path}: {mse_loss.item()}")

# 推理数据集中的每张图像
lr_dir = '../data/Set5_LR'
hr_dir = '../data/Set5'
image_names = [f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]

for img_name in image_names:
    lr_image_path = os.path.join(lr_dir, img_name)
    hr_image_path = os.path.join(hr_dir, img_name)
    infer_image(lr_image_path, hr_image_path)

print('Finished Inference')
