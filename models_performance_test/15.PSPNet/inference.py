import os
import torch
import torch.nn as nn
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F_transforms
from torchvision.models import resnet50
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=max(size, 2)),  # 确保池化结果至少是 2x2
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for size in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
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
        backbone = resnet50(pretrained=False)

        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path)
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
    mask[(mask < 0) | (mask >= 21)] = -1  # 将超出范围的值标记为无效值
    return mask


# 加载模型
def load_model(model_path, num_classes=21, device="cuda"):
    model = PSPNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path,weights_only=True, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


# 计算 IoU
def calculate_iou(preds, targets, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union == 0:
            ious.append(float('nan'))  # 忽略该类别
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)  # 计算所有类的平均 IoU


# 计算像素精度
def calculate_pixel_accuracy(preds, targets):
    valid = targets != -1  # 忽略无效值
    correct = (preds == targets) & valid
    accuracy = correct.sum() / valid.sum()
    return accuracy


# 推理函数
def evaluate_model(model, data_loader, device="cuda", num_classes=21):
    model.eval()
    iou_scores = []
    pixel_accuracies = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)

            # 模型前向推理
            outputs = model(images)  # [batch_size, num_classes, H, W]
            preds = torch.argmax(outputs, dim=1)  # [batch_size, H, W]

            # 计算指标
            for pred, mask in zip(preds, masks):
                iou = calculate_iou(pred.cpu().numpy(), mask.cpu().numpy(), num_classes)
                pixel_acc = calculate_pixel_accuracy(pred.cpu().numpy(), mask.cpu().numpy())
                iou_scores.append(iou)
                pixel_accuracies.append(pixel_acc)

    mean_iou = np.nanmean(iou_scores)  # 忽略 NaN
    mean_pixel_accuracy = np.mean(pixel_accuracies)
    return mean_iou, mean_pixel_accuracy


# 主推理逻辑
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "pspnet_model.pth"

    # 数据加载与预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = VOCSegmentation(
        root='../data',
        year='2007',
        image_set='val',
        download=False,
        transform=transform,
        target_transform=mask_transform
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # 加载模型
    model = load_model(model_path, num_classes=21, device=device)

    # 数据集评估
    mean_iou, mean_pixel_accuracy = evaluate_model(model, data_loader, device=device, num_classes=21)

    # 打印性能指标
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Pixel Accuracy: {mean_pixel_accuracy:.4f}")


if __name__ == "__main__":
    main()
