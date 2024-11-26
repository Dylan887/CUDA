import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import functional as F_transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.resnet import ResNet, Bottleneck
from torch.utils.data import DataLoader
from torchvision import transforms


# 自定义标签处理函数
def mask_transform(mask):
    mask = mask.resize((256, 256))  # 调整标签大小
    mask = F_transforms.pil_to_tensor(mask).squeeze(0)
    mask = mask.long()
    mask[(mask < 0) | (mask >= 21)] = -1  # 将超出范围的值标记为无效值
    return mask


# 自定义 ResNet50 加载函数
def load_custom_resnet50(weights_path=None, device="cuda"):
    # 构建 ResNet50 Backbone
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],  # ResNet-50 配置
        replace_stride_with_dilation=[False, True, True]
    )
    if weights_path:
        print(f"Loading ResNet50 backbone weights from {weights_path}")
        state_dict = torch.load(weights_path, weights_only=True,map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # 删除最后的全连接层及分类相关层，保留到最后的卷积层
    model = nn.Sequential(*list(model.children())[:-2])
    return model.to(device)


# 自定义 DeepLabV3 模型加载函数
def load_deeplab_model(num_classes=21, deeplab_pretrained_path=None, resnet_pretrained_path=None, device="cuda"):
    # 加载自定义的 ResNet50 Backbone
    backbone = load_custom_resnet50(weights_path=resnet_pretrained_path, device=device)

    # 构建 DeepLab 模型
    class DeepLabV3(nn.Module):
        def __init__(self, backbone, num_classes):
            super(DeepLabV3, self).__init__()
            self.backbone = backbone
            self.classifier = DeepLabHead(2048, num_classes)  # 2048是 ResNet 的输出通道数

        def forward(self, x):
            features = self.backbone(x)  # 获取特征图
            return {"out": self.classifier(features)}

    model = DeepLabV3(backbone, num_classes)

    # 加载 DeepLab 的预训练权重
    if deeplab_pretrained_path:
        print(f"Loading DeepLab pretrained weights from {deeplab_pretrained_path}")
        state_dict = torch.load(deeplab_pretrained_path, weights_only=True,map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    return model


# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs, device="cuda"):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            # 前向传播调试
            features = model.backbone(images)
          
            outputs = model.classifier(features)
         
            # 前向传播
            outputs = model(images)['out']  # DeepLab 返回一个字典
            outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)  # 上采样

            # 计算损失
            loss = criterion(outputs, masks)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 主训练逻辑
def main():
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 21
    num_epochs = 3
    batch_size = 4
    learning_rate = 0.001

    # 数据预处理与加载
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
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 手动下载的预训练权重路径
    resnet_pretrained_path = "../pre_weights/resnet50-0676ba61.pth"
    deeplab_pretrained_path = "../pre_weights/deeplabv3_resnet50_coco-cd0a2569.pth"

    # 加载模型
    model = load_deeplab_model(
        num_classes=num_classes,
        deeplab_pretrained_path=deeplab_pretrained_path,
        resnet_pretrained_path=resnet_pretrained_path,
        device=device
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略无效标签
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs, device=device)

    # 保存模型
    torch.save(model.state_dict(), "deeplab_model.pth")
    print("Model training complete and saved to deeplab_model.pth")


if __name__ == "__main__":
    main()
