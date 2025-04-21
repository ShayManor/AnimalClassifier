import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from AnimalClassifier import AnimalClassifier
from AnimalDataset import AnimalDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = AnimalDataset('train.csv', 'Animals', transform)
validation = AnimalDataset('test.csv', 'Animals', transform)
validation_dataloader = DataLoader(validation, batch_size=16, shuffle=True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
num_classes = 3
CLASS_LIST = ["dog", "cat", "snake"]
model = AnimalClassifier(3, INPUT_SIZE, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
EPOCHS = 40
all_predictions, all_labels = [], []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(dataloader):.4f}')

    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in validation_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions) * 100

    print(f'Epoch [{epoch + 1}/{EPOCHS}]')
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print('-----------------------------')

print('Training finished!')
torch.save(model.state_dict(), 'weights.pth')