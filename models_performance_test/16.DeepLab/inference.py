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


# 加载模型函数
def load_deeplab_model(num_classes=21, deeplab_pretrained_path=None, resnet_pretrained_path=None, device="cuda"):
    # 构建 ResNet50 Backbone
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],  # ResNet-50 配置
        replace_stride_with_dilation=[False, True, True]
    )
    if resnet_pretrained_path:
        print(f"Loading ResNet50 backbone weights from {resnet_pretrained_path}")
        state_dict = torch.load(resnet_pretrained_path, weights_only=True,map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # 删除最后的全连接层及分类相关层，保留到最后的卷积层
    backbone = nn.Sequential(*list(model.children())[:-2])

    # 构建 DeepLab 模型
    class DeepLabV3(nn.Module):
        def __init__(self, backbone, num_classes):
            super(DeepLabV3, self).__init__()
            self.backbone = backbone
            self.classifier = DeepLabHead(2048, num_classes)

        def forward(self, x):
            features = self.backbone(x)
            return {"out": self.classifier(features)}

    model = DeepLabV3(backbone, num_classes)

    # 加载 DeepLab 的预训练权重
    if deeplab_pretrained_path:
        print(f"Loading DeepLab pretrained weights from {deeplab_pretrained_path}")
        state_dict = torch.load(deeplab_pretrained_path,weights_only=True, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


# 推理函数并计算指标
def evaluate(model, data_loader, device="cuda"):
    intersection = torch.zeros(21).to(device)  # 交集
    union = torch.zeros(21).to(device)         # 并集
    total_correct = 0                          # 总正确像素
    total_pixels = 0                           # 总像素

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)

            # 模型推理
            outputs = model(images)['out']
            outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)  # 调整尺寸
            predictions = torch.argmax(outputs, dim=1)  # 获取预测类别

            # 计算指标
            for cls in range(21):
                pred_cls = predictions == cls
                true_cls = masks == cls
                intersection[cls] += torch.sum(pred_cls & true_cls)
                union[cls] += torch.sum(pred_cls | true_cls)

            total_correct += torch.sum(predictions == masks).item()
            total_pixels += masks.numel()

    # 计算 mIoU 和像素准确率
    iou = intersection / (union + 1e-6)
    miou = torch.mean(iou).item()
    pixel_accuracy = total_correct / total_pixels
    return miou, pixel_accuracy



# 主推理逻辑
def main_inference():
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 21
    batch_size = 4
    resnet_pretrained_path = "../pre_weights/resnet50-0676ba61.pth"
    deeplab_pretrained_path = "deeplab_model.pth"  # 已训练的模型权重

    # 数据预处理与加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = VOCSegmentation(
        root='../data',
        year='2007',
        image_set='val',  # 验证集
        download=False,
        transform=transform,
        target_transform=mask_transform
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = load_deeplab_model(
        num_classes=num_classes,
        deeplab_pretrained_path=deeplab_pretrained_path,
        resnet_pretrained_path=resnet_pretrained_path,
        device=device
    )

    # 评估
    miou, pixel_accuracy = evaluate(model, data_loader, device=device)
    print(f"Mean IoU: {miou:.4f}")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")


if __name__ == "__main__":
    main_inference()
