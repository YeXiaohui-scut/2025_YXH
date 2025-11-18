import torch
from torchvision import datasets, transforms
from my_model_library import VAE, DistortionLayer

class Stage2Trainer:
    def __init__(self, model, optimizer, criterion, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader

    def train(self, epochs):
        for epoch in range(epochs):
            for data in self.dataloader:
                inputs, targets = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                print(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CocoDetection(root='path/to/coco', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    trainer = Stage2Trainer(model, optimizer, criterion, dataloader)
    trainer.train(epochs=10)