import os
import torch
import torch.nn as nn
from torchvision import transforms
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
# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图像调整为适合 U-Net 的输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载 U-Net 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load('unet_set5.pth',weights_only=True))  # 加载训练好的模型权重
model.eval()

# 推理函数
def infer_image(lr_image_path, hr_image_path):
    lr_image = Image.open(lr_image_path).convert('RGB')
    hr_image = Image.open(hr_image_path).convert('RGB')
    input_tensor = transform(lr_image).unsqueeze(0).to(device)
    hr_tensor = transform(hr_image).unsqueeze(0).to(device)  # 确保高分辨率图像也在相同的设备上
    
    with torch.no_grad():
        output = model(input_tensor)
    
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
