import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class ChartDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.data = {f.replace('.png', '.json'): json.load(open(os.path.join(json_dir, f.replace('.png', '.json')))) for f in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)  # Convert to PIL for torchvision
        data = self.data[image_name.replace('.png', '.json')]
        prices = torch.tensor(data['prices'], dtype=torch.float32)
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
        self.fc1 = nn.Linear(32 * 64 * 64, 256)  # Adjust for image size (e.g., 256x256)
        self.fc2 = nn.Linear(256, 101)  # Output 101 prices (days 0–100)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Usage
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load your dataset (images, prices from labeled data)
dataset = ChartDataset(images=[chart1, chart2, ...], prices=[prices1, prices2, ...], transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = ChartCNN()
criterion = nn.MSELoss()  # Mean Squared Error for price prediction
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified, run for 10–20 epochs)
for epoch in range(10):
    for images, prices in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, prices)
        loss.backward()
        optimizer.step()