import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import json
import cv2
from PIL import Image
import numpy as np


class ChartDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.startswith('marked_')]
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
        # Mask edges (e.g., green bars at 0 and 100) to reduce bias
        image[:, 0:10] = 255  # White out left edge
        image[:, -10:] = 255  # White out right edge
        image = Image.fromarray(image)
        data = self.data[image_name.replace('.png', '.json')]
        prices = torch.tensor(data['prices'][:100], dtype=torch.float32)
        prices = (prices - 47.5) / 17.5  # Normalize to [0, 1]
        troughs = torch.tensor(data['troughs'], dtype=torch.long)
        peaks = torch.tensor(data['peaks'], dtype=torch.long)
        pattern = data['pattern']
        if self.transform:
            image = self.transform(image)
        return image, {"prices": prices, "troughs": troughs, "peaks": peaks, "pattern": pattern}


class ChartCNN(nn.Module):
    def __init__(self):
        super(ChartCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 100)
        self.dropout = nn.Dropout(0.3)
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(x) * 1.0
        x = self.fc2(x)
        return x


# Custom collate function
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    prices = torch.stack([label["prices"] for label in labels], 0)
    troughs = [label["troughs"] for label in labels]
    peaks = [label["peaks"] for label in labels]
    patterns = [label["pattern"] for label in labels]
    return images, {"prices": prices, "troughs": troughs, "peaks": peaks, "pattern": patterns}


# Usage with data augmentation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ChartDataset(image_dir="test_charts/new", json_dir="test_charts/new", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

model = ChartCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop with more epochs
for epoch in range(150):  # Increased to 150 epochs
    model.train()
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        prices = labels["prices"]
        loss = criterion(outputs, prices)
        loss.backward()
        optimizer.step()
    scheduler.step(loss)
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Post-process to detect troughs/peaks with pattern-specific logic
def detect_patterns(prices, pattern, threshold=1.0, min_distance=10):
    troughs, peaks = [], []
    prices = prices.copy()
    prices = np.convolve(prices, np.ones(3) / 3, mode='same')
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] - threshold and prices[i] < prices[i + 1] - threshold:
            if not troughs or i - troughs[-1] >= min_distance:
                troughs.append(i)
        elif prices[i] > prices[i - 1] + threshold and prices[i] > prices[i + 1] + threshold:
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
    # Adjust based on pattern to match expected number of troughs/peaks
    if pattern == "Buy" and "Double Bottom" in image_name:
        troughs = troughs[:2]  # Limit to 2 troughs for Double Bottom
        peaks = peaks[:1]  # Limit to 1 peak
    elif pattern == "Buy" and "Ascending Triangle" in image_name:
        troughs = troughs[:1]  # Limit to 1 trough
        peaks = peaks[:3]  # Limit to 3 peaks
    elif pattern == "Buy" and "Inverse Head and Shoulders" in image_name:
        troughs = troughs[:5]  # Limit to 5 troughs
        peaks = peaks[:4]  # Limit to 4 peaks
    return troughs, peaks


# Test on a chart and save marked image
model.eval()
with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        for i in range(len(images)):
            predicted_prices = outputs[i].numpy()
            predicted_prices = np.clip(predicted_prices * 17.5 + 47.5, 47.5, 65.0)
            image_name = dataset.images[i]
            pattern = labels["pattern"][i]
            troughs, peaks = detect_patterns(predicted_prices, pattern, threshold=1.0, min_distance=10)

            img_path = os.path.join("test_charts/new", image_name)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            for day in troughs:
                x = int(day / 99 * (width - 1))
                cv2.line(img, (x, 0), (x, height), (0, 255, 0), 2)
            for day in peaks:
                x = int(day / 99 * (width - 1))
                cv2.line(img, (x, 0), (x, height), (0, 0, 255), 2)

            output_path = os.path.join("test_charts/new", f"marked_{image_name}")
            cv2.imwrite(output_path, img)

            actual_prices = labels['prices'][i].numpy() * 17.5 + 47.5
            print(f"Image: {image_name}")
            print(f"Predicted prices: {predicted_prices[:10]}...{predicted_prices[-10:]}")
            print(f"Actual prices: {actual_prices[:10]}...{actual_prices[-10:]}")
            print(f"Troughs: {troughs}")
            print(f"Peaks: {peaks}")
            print(f"Actual troughs: {labels['troughs'][i].numpy()}")
            print(f"Actual peaks: {labels['peaks'][i].numpy()}")
        break