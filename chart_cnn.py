import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import json
import cv2
from PIL import Image


class ChartDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.data = {f.replace('.png', '.json'): json.load(open(os.path.join(json_dir, f.replace('.png', '.json')))) for
                     f in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = Image.fromarray(image)  # Convert to PIL for torchvision
        data = self.data[image_name.replace('.png', '.json')]
        prices = torch.tensor(data['prices'][:100], dtype=torch.float32)  # Ensure 100 elements
        troughs = torch.tensor(data['troughs'], dtype=torch.long)
        peaks = torch.tensor(data['peaks'], dtype=torch.long)
        pattern = data['pattern']
        if self.transform:
            image = self.transform(image)
        return image, {"prices": prices, "troughs": troughs, "peaks": peaks, "pattern": pattern}


class ChartCNN(nn.Module):
    def __init__(self):
        super(ChartCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Added deeper layer
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Adjusted for deeper architecture
        self.fc2 = nn.Linear(512, 100)  # Output 100 prices (days 0–99)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))  # Added pooling after conv3
        x = x.view(-1, 64 * 32 * 32)  # Adjusted view for deeper architecture
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Custom collate function to handle variable-length tensors
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)  # Stack images (fixed size [batch, 1, 256, 256])

    # Stack fixed-size prices
    prices = torch.stack([label["prices"] for label in labels], 0)

    # Convert variable-length troughs and peaks to lists
    troughs = [label["troughs"] for label in labels]  # Keep as list of tensors
    peaks = [label["peaks"] for label in labels]  # Keep as list of tensors
    patterns = [label["pattern"] for label in labels]  # Keep as list of strings

    return images, {"prices": prices, "troughs": troughs, "peaks": peaks, "pattern": patterns}


# Usage with data augmentation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.RandomRotation(10),  # Add rotation for augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load your dataset from test_charts/new
dataset = ChartDataset(image_dir="test_charts/new", json_dir="test_charts/new", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

model = ChartCNN()
criterion = nn.MSELoss()  # Mean Squared Error for price prediction
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with more epochs
for epoch in range(20):  # Increased to 20 epochs
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)  # Predict prices (shape [batch, 100])
        prices = labels["prices"]  # Ground-truth prices (shape [batch, 100])
        loss = criterion(outputs, prices)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")  # Add loss printing for monitoring


# Post-process to detect troughs/peaks
def detect_patterns(prices):
    troughs, peaks = [], []
    for i in range(1, len(prices) - 1):  # No need for .numpy() since prices is already numpy-compatible
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            troughs.append(i)
        elif prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            peaks.append(i)
    return troughs, peaks


# Test on a chart and save marked image
model.eval()
import matplotlib.pyplot as plt

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        for i in range(len(images)):  # Process each image in the batch
            predicted_prices = outputs[i].numpy()
            troughs, peaks = detect_patterns(predicted_prices)

            # Load original image for marking
            image_name = dataset.images[i]
            img_path = os.path.join("test_charts/new", image_name)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            # Mark troughs (green) and peaks (red)
            for day in troughs:
                x = int(day / 99 * (width - 1))  # Adjust for 100 days (0–99)
                cv2.line(img, (x, 0), (x, height), (0, 255, 0), 2)  # Green for troughs
            for day in peaks:
                x = int(day / 99 * (width - 1))  # Adjust for 100 days (0–99)
                cv2.line(img, (x, 0), (x, height), (0, 0, 255), 2)  # Red for peaks

            # Save marked image
            output_path = os.path.join("test_charts/new", f"marked_{image_name}")
            cv2.imwrite(output_path, img)

            print(f"Image: {image_name}")
            print(f"Predicted prices: {predicted_prices[:10]}...{predicted_prices[-10:]}")
            print(f"Actual prices: {labels['prices'][i][:10].numpy()}...{labels['prices'][i][-10:].numpy()}")
            print(f"Troughs: {troughs}")
            print(f"Peaks: {peaks}")
            print(f"Actual troughs: {labels['troughs'][i].numpy()}")
            print(f"Actual peaks: {labels['peaks'][i].numpy()}")
        break  # Test first batch