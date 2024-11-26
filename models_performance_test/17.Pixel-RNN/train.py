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

# 数据加载与预处理
def load_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
    ])

    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            # 将输入值映射到 [0, 255] 的整数值范围
            targets = (images * 255).clamp(0, 255).long()
            targets = targets.squeeze(1)  # 去掉单通道维度
            
            # 检查目标值范围
            assert targets.min() >= 0 and targets.max() <= 255, "Target values out of range!"

            # 前向传播
            outputs = model(targets)
            outputs = outputs.permute(0, 3, 1, 2)  # 调整维度为 (batch_size, num_classes, height, width)

            # 计算损失
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 3
    learning_rate = 0.001
    input_dim = 16  # 嵌入维度
    hidden_dim = 64  # RNN 隐藏层维度
    num_classes = 256  # 灰度值范围

    # 加载数据
    train_loader = load_data(batch_size)

    # 初始化模型、损失函数、优化器
    model = PixelRNN(input_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, num_epochs)

    # 保存模型
    torch.save(model.state_dict(), "pixel_rnn_mnist.pth")
    print("Model training complete and saved to pixel_rnn_mnist.pth")

if __name__ == "__main__":
    main()
