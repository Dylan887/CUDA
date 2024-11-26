import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# 定义 PixelRNN 模型
class PixelRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(PixelRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_classes, input_dim)
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 嵌入输入（独热编码 -> 嵌入）
        x = self.embedding(x.long())  # 输入为整数类型
        x = x.view(x.size(0), -1, x.size(-1))  # 展平图像
        # 经过 RNN
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        # 全连接输出
        out = self.fc(out)
        return out.view(-1, 28, 28, 256)  # 输出与输入图像大小一致
# 定义计算像素准确率的函数
def calculate_pixel_accuracy(predictions, targets):
    """
    计算像素准确率（Pixel Accuracy）。
    :param predictions: 模型输出的预测值 (B, H, W)
    :param targets: 实际目标值 (B, H, W)
    :return: 像素准确率
    """
    correct = (predictions == targets).sum().item()
    total = targets.numel()  # 总像素数
    return correct / total

# 定义推理函数
def infer_model(model, test_loader, device):
    model.eval()
    total_pixel_accuracy = 0

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            targets = (images * 255).clamp(0, 255).long()
            targets = targets.squeeze(1)  # 去掉单通道维度
            
            # 前向传播
            outputs = model(targets)
            outputs = outputs.permute(0, 3, 1, 2)  # 调整维度为 (batch_size, num_classes, height, width)

            # 获取每个像素的预测类别
            predictions = torch.argmax(outputs, dim=1)

            # 计算像素准确率
            pixel_accuracy = calculate_pixel_accuracy(predictions, targets)
            total_pixel_accuracy += pixel_accuracy

    # 返回平均像素准确率
    avg_pixel_accuracy = total_pixel_accuracy / len(test_loader)
    return avg_pixel_accuracy

# 主推理逻辑
def main_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    input_dim = 16  # 嵌入维度
    hidden_dim = 64  # RNN 隐藏层维度
    num_classes = 256  # 灰度值范围

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
    ])
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = PixelRNN(input_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load("pixel_rnn_mnist.pth", map_location=device))

    # 推理并计算指标
    avg_pixel_accuracy = infer_model(model, test_loader, device)
    print(f"Average Pixel Accuracy: {avg_pixel_accuracy:.4f}")

if __name__ == "__main__":
    main_inference()
