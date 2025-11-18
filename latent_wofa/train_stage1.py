# Complete Training Script for Stage I

# Import necessary modules
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import MyDataset  # Assuming `MyDataset` is your dataset class
from models import Stage1Model  # Assuming your model is defined in models

# Training settings
epochs = 100
batch_size = 16
learning_rate = 0.0001

# Data preparation
train_dataset = MyDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, loss function, optimizer
model = Stage1Model().to('cuda')  # Assuming using GPU
criterion = torch.nn.CrossEntropyLoss()  # Replace with your loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')