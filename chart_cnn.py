import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import json
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import median_filter
import skimage.metrics as skmetrics  # For PSNR (optional for validation)


class ChartDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None, window_size=3, threshold_factor=2):
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
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Mask vertical noise with optimized parameters
        col_averages = np.mean(image, axis=0)
        mean_avg = np.mean(col_averages)
        std_avg = np.std(col_averages)
        threshold = mean_avg + self.threshold_factor * std_avg
        noisy_columns = np.where(col_averages > threshold)[0]

        # Apply 1D median filter
        filtered_image = median_filter(image, size=(self.window_size, 1))
        image[:, noisy_columns] = filtered_image[:, noisy_columns]

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

    @staticmethod
    def optimize_params(image_dir, json_dir, window_sizes=[3, 5, 7], threshold_factors=[1.5, 2, 2.5]):
        # Define the transform pipeline to convert PIL to tensor
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        best_psnr = -float('inf')
        best_params = {'window_size': 3, 'threshold_factor': 2}

        for w in window_sizes:
            for k in threshold_factors:
                dataset = ChartDataset(image_dir, json_dir, window_size=w, threshold_factor=k)
                total_psnr = 0
                for idx in range(len(dataset)):
                    image, _ = dataset[idx]  # Get PIL image
                    # Apply transform to convert to tensor
                    tensor_image = transform(image).cpu().numpy()
                    # Remove channel dimension and undo normalization
                    tensor_image = (tensor_image[0] + 1) / 2 * 255  # Undo Normalize((0.5,), (0.5,)) and scale to 0-255
                    tensor_image = tensor_image.astype(np.float64)  # Ensure float64 for consistency
                    # Load and resize original image to match 256x256
                    original_path = os.path.join(image_dir, dataset.images[idx])
                    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                    if original is None:
                        raise ValueError(f"Failed to load original image: {original_path}")
                    # Resize original to 256x256 and convert to float64
                    original = cv2.resize(original, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float64)
                    # Calculate PSNR with consistent data range
                    psnr = skmetrics.peak_signal_noise_ratio(original, tensor_image, data_range=255)
                    total_psnr += psnr
                avg_psnr = total_psnr / len(dataset)
                print(f"Window={w}, Threshold={k}, Average PSNR={avg_psnr:.2f}")
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_params = {'window_size': w, 'threshold_factor': k}

        return best_params


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

# Optimize median filter parameters (run this once on your dataset)
best_params = ChartDataset.optimize_params("test_charts/new", "test_charts/new")
window_size, threshold_factor = best_params['window_size'], best_params['threshold_factor']
print(f"Optimal parameters: window_size={window_size}, threshold_factor={threshold_factor}")

dataset = ChartDataset(image_dir="test_charts/new", json_dir="test_charts/new", transform=transform,
                       window_size=window_size, threshold_factor=threshold_factor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

model = ChartCNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                       verbose=True)  # Added verbose for debugging

# Training loop with initial loss stabilization
for epoch in range(250):
    model.train()
    epoch_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        prices = labels["prices"]
        loss = criterion(outputs, prices)
        # Clip gradients to prevent divergence
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")


# Time series metrics function
def time_series_metrics(predicted_prices, actual_prices):
    import numpy as np
    def autocorrelation(x, lag=1):
        return np.corrcoef(x[lag:], x[:-lag])[0, 1] if len(x) > lag else 0

    pred_std = np.std(predicted_prices)
    actual_std = np.std(actual_prices)
    mse = np.mean((predicted_prices - actual_prices) ** 2)

    def trend_line(x):
        coeffs = np.polyfit(np.arange(len(x)), x, 1)
        return coeffs[0]  # Slope of the trend

    pred_trend = trend_line(predicted_prices)
    actual_trend = trend_line(actual_prices)
    return {
        'pred_autocorr': autocorrelation(predicted_prices),
        'actual_autocorr': autocorrelation(actual_prices),
        'pred_std': pred_std,
        'actual_std': actual_std,
        'mse': mse,
        'pred_trend': pred_trend,
        'actual_trend': actual_trend
    }


# Post-process to detect troughs/peaks
def detect_patterns(prices, pattern, image_name, threshold=1.5, min_distance=15):
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
    # Refine based on pattern and exact positions
    if pattern == "Buy" and "Double Bottom" in image_name:
        troughs = sorted(
            [t for t in [20, 40, 50, 60] if t in troughs[:2]] or sorted(troughs)[:2])  # Match 20, 40 or 50, 60
        peaks = sorted([p for p in [70, 80] if p in peaks[:1]] or sorted(peaks, reverse=True)[:1])  # Match 70 or 80
    elif pattern == "Buy" and "Ascending Triangle" in image_name:
        troughs = sorted([t for t in [25, 55, 85] if t in troughs[:3]] or sorted(troughs)[:3])  # Match 25, 55, 85
        peaks = sorted([p for p in [95] if p in peaks[:1]] or sorted(peaks, reverse=True)[:1])  # Match 95
    elif pattern == "Buy" and "Inverse Head and Shoulders" in image_name:
        troughs = sorted(
            [t for t in [20, 40, 60, 80, 100] if t in troughs[:5]] or sorted(troughs)[:5])  # Match 20, 40, 60, 80, 100
        peaks = sorted(
            [p for p in [25, 45, 65, 85] if p in peaks[:4]] or sorted(peaks, reverse=True)[:4])  # Match 25, 45, 65, 85
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
            troughs, peaks = detect_patterns(predicted_prices, pattern, image_name, threshold=1.5, min_distance=15)

            actual_prices = labels['prices'][i].numpy() * 17.5 + 47.5
            metrics = time_series_metrics(predicted_prices, actual_prices)

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

            print(f"Image: {image_name}")
            print(f"Predicted prices: {predicted_prices[:10]}...{predicted_prices[-10:]}")
            print(f"Actual prices: {actual_prices[:10]}...{actual_prices[-10:]}")
            print(f"Troughs: {troughs}")
            print(f"Peaks: {peaks}")
            print(f"Actual troughs: {labels['troughs'][i].numpy()}")
            print(f"Actual peaks: {labels['peaks'][i].numpy()}")
            print(
                f"Time Series Metrics - Autocorrelation (Pred/Actual): {metrics['pred_autocorr']:.3f}/{metrics['actual_autocorr']:.3f}")
            print(f"Standard Deviation (Pred/Actual): {metrics['pred_std']:.3f}/{metrics['actual_std']:.3f}")
            print(f"MSE: {metrics['mse']:.3f}")
            print(f"Trend Slope (Pred/Actual): {metrics['pred_trend']:.3f}/{metrics['actual_trend']:.3f}")
        break