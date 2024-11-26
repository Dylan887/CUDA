import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize MNIST images (28x28) to (224x224) for ResNet-V2-50
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert single channel to 3 channels
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset normalization parameters
])

# Load the MNIST dataset
train_dataset = torch.utils.data.Subset(datasets.MNIST(root='../data', train=True, download=False, transform=transform), range(1000))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

# Load ResNet-V2-50 model from pretrainedmodels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)  #
model.fc = nn.Linear(model.fc.in_features, 10)  #
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 3
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print loss every 100 batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'resnetv2_50_mnist.pth')  # Save model state dict
