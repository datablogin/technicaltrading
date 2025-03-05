import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
import cv2
import numpy as np
from PIL import Image

# Import preprocessing functions
from preprocessing import mask_vertical_noise, segment_chart_region, extract_chart_line


class ChartDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None, window_size=3, threshold_factor=3):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.startswith('marked_')]
        self.data = {f.replace('.png', '.json'): json.load(open(os.path.join(json_dir, f.replace('.png', '.json')))) for
                     f in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Enhanced preprocessing
        # 1. Segment chart region
        chart_box = segment_chart_region(image)

        # 2. Extract chart line
        chart_line = extract_chart_line(image, chart_box)

        # 3. Mask vertical noise
        if len(chart_line.shape) == 2:  # If grayscale
            chart_line = mask_vertical_noise(chart_line, self.window_size, self.threshold_factor)

        # Convert to PIL for transforms
        if len(chart_line.shape) == 2:  # If grayscale
            chart_line = Image.fromarray(chart_line)
        else:
            chart_line = Image.fromarray(cv2.cvtColor(chart_line, cv2.COLOR_BGR2RGB))

        # Load labels
        data = self.data[image_name.replace('.png', '.json')]
        prices = torch.tensor(data['prices'][:100], dtype=torch.float32)
        prices = (prices - 47.5) / 17.5  # Normalize to [0, 1]
        troughs = torch.tensor(data['troughs'], dtype=torch.long)
        peaks = torch.tensor(data['peaks'], dtype=torch.long)
        pattern = data['pattern']

        if self.transform:
            chart_line = self.transform(chart_line)

        return chart_line, {"prices": prices, "troughs": troughs, "peaks": peaks, "pattern": pattern}


# Keep your ChartCNN class as is
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
        x = self.fc2(x)
        return x