import torch
from torchvision import transforms
from PIL import Image
import os

from train import SPADEGenerator, VOCDataset

# 设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data/VOC2007"
batch_size = 8

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# 加载数据
dataset = VOCDataset(root_dir, transform=transform, target_transform=target_transform)
subset_size = 50  # 使用部分数据进行测试
dataset, _ = torch.utils.data.random_split(dataset, [subset_size, len(dataset) - subset_size])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 加载模型
model = SPADEGenerator(label_nc=21).to(device)
model.load_state_dict(torch.load("spade_generator.pth"))
model.eval()

# 推理
with torch.no_grad():
    for i, (images, masks) in enumerate(dataloader):
        masks = masks.to(device)
        outputs = model(masks)

        # 保存生成的图像
        for j in range(outputs.size(0)):
            output = (outputs[j].cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255
            output = output.astype("uint8")
            output_img = Image.fromarray(output)
            output_img.save(f"output_{i*batch_size+j}.png")

print("Inference completed!")
