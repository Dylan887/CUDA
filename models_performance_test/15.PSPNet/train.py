import os
import torch
import torch.nn as nn
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F_transforms
from torchvision.models import resnet50
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([  # 定义多个不同尺寸的池化层
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=max(size, 2)),  # 确保池化结果至少是 2x2
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
        self.bottleneck = nn.Sequential(  # 聚合后的卷积层
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.size()[2:]  # 原始输入的空间大小
        pooled = [x]  # 添加原始特征
        for stage in self.stages:
            pooled.append(F.interpolate(stage(x), size=size, mode='bilinear', align_corners=False))
        out = torch.cat(pooled, dim=1)
        return self.bottleneck(out)


class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super(PSPNet, self).__init__()
        backbone = resnet50(pretrained=False)  # 依然不从官方库自动加载权重

        if pretrained_path:  # 如果提供了本地的预训练权重文件路径
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, weights_only=True,map_location='cpu')  # 加载权重
            backbone.load_state_dict(state_dict)

        # ResNet backbone
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(in_channels=2048, out_channels=512, pool_sizes=[1, 2, 3, 6])

        # Final classification head
        self.final = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        size = x.size()[2:]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ppm(x)
        x = self.final(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# 自定义标签处理函数
def mask_transform(mask):
    mask = mask.resize((256, 256))  # 调整标签大小与输入一致
    mask = F_transforms.pil_to_tensor(mask).squeeze(0)
    mask = mask.long()

    # 将超出范围的值标记为无效值
    mask[(mask < 0) | (mask >= 21)] = -1
    return mask


# 主函数
def main():
    # 数据加载与预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = VOCSegmentation(
        root='../data',
        year='2007',
        image_set='train',
        download=False,
        transform=transform,
        target_transform=mask_transform
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 手动下载的预训练模型路径
    pretrained_path = "../pre_weights/resnet50-0676ba61.pth"

    # 模型定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSPNet(num_classes=21, pretrained_path=pretrained_path).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):  # 假设训练 3 个 epoch
        model.train()
        total_loss = 0
        for batch_idx, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}")

    # 保存模型
    model_save_path = "pspnet_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model training complete and saved to {model_save_path}.")


if __name__ == "__main__":
    main()
