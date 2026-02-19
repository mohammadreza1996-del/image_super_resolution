import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import SRCNN
from src.dataset import SRDataset
import torch.nn as nn
import torch.optim as optim

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset و Dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = SRDataset(root_dir='data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


model = SRCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 50
for epoch in range(num_epochs):
    for imgs in train_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


torch.save(model.state_dict(), 'models/srcnn.pth')
