import os
import torch
import torch.nn as nn
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F_transforms
import numpy as np
# 定义标准的 ICNet 模型
class ICNet(nn.Module):
    def __init__(self, num_classes=21):
        super(ICNet, self).__init__()
        # 定义多个卷积分支，分别处理不同尺度的特征
        self.low_res_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 下采样
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )
        self.middle_res_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )
        self.high_res_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )
        
        # 融合特征
        self.fuse = nn.Conv2d(3 * num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        # 低分辨率分支
        low_res = self.low_res_branch(x)
        low_res = nn.functional.interpolate(low_res, size=(128, 128), mode='bilinear', align_corners=False)
        
        # 中分辨率分支
        middle_res = self.middle_res_branch(x)
        middle_res = nn.functional.interpolate(middle_res, size=(128, 128), mode='bilinear', align_corners=False)
        
        # 高分辨率分支
        high_res = self.high_res_branch(x)
        
        # 融合各个分支
        fused = torch.cat((low_res, middle_res, high_res), dim=1)
        output = self.fuse(fused)
        
        return output

# 数据加载器中的标签处理
def mask_transform(mask):
    mask = mask.resize((128, 128))
    mask = F_transforms.pil_to_tensor(mask).squeeze(0)
    mask = mask.long()

    # 将标签中不在 0 到 20 范围内的值标记为无效值
    mask[(mask < 0) | (mask >= 21)] = -1
    return mask

# 训练函数
def train(model, device, data_loader, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, masks) in enumerate(data_loader):
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, masks)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader)}")

# 推理与评估函数
def infer_and_evaluate(model, device, data_loader):
    model.eval()
    iou_scores = []
    pixel_accuracies = []

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            # 推理
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            # 计算指标
            for pred, target in zip(predicted, masks):
                iou = compute_iou(pred.cpu().numpy(), target.cpu().numpy())
                pixel_acc = compute_pixel_accuracy(pred.cpu().numpy(), target.cpu().numpy())

                iou_scores.append(iou)
                pixel_accuracies.append(pixel_acc)

    avg_iou = np.nanmean(iou_scores)
    avg_pixel_acc = np.mean(pixel_accuracies)
    return avg_iou, avg_pixel_acc

# 计算 IoU
def compute_iou(predicted, target, num_classes=21):
    iou_scores = []
    for cls in range(num_classes):
        pred_mask = (predicted == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()

        if union == 0:
            iou_scores.append(float('nan'))  # 忽略该类
        else:
            iou_scores.append(intersection.item() / union.item())
    return np.nanmean(iou_scores)

# 计算像素准确率
def compute_pixel_accuracy(predicted, target):
    valid = (target != -1)  # 忽略无效像素
    correct = (predicted[valid] == target[valid]).sum()
    total = valid.sum()
    return correct.item() / total.item()

# 主函数
def main():
    # 数据加载与预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
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

    # 模型定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ICNet(num_classes=21).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, device, data_loader, criterion, optimizer)

    # 保存模型
    torch.save(model.state_dict(), "icnet_model.pth")
    print("Model training complete and saved.")

if __name__ == "__main__":
    main()
