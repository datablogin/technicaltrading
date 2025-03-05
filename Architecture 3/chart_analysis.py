import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image


# Import your modules
from model import ChartCNN, ChartDataset
from pattern_detection import detect_technical_patterns
from preprocessing import segment_chart_region, extract_chart_line, mask_vertical_noise

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced data augmentation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.RandomRotation(5),  # Reduced rotation
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),  # Small translations
        scale=(0.95, 1.05),  # Small scaling
        fill=255  # Fill with white background
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Custom collate function
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0).to(device)
    prices = torch.stack([label["prices"] for label in labels], 0).to(device)
    troughs = [label["troughs"] for label in labels]
    peaks = [label["peaks"] for label in labels]
    patterns = [label["pattern"] for label in labels]
    return images, {"prices": prices, "troughs": troughs, "peaks": peaks, "pattern": patterns}


def train_model(image_dir, json_dir, num_epochs=10):  # Reduced epochs for testing
    """Train the CNN model"""
    # Create dataset and dataloader
    dataset = ChartDataset(image_dir=image_dir, json_dir=json_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = ChartCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            prices = labels["prices"]
            loss = criterion(outputs, prices)
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    # Save model
    torch.save(model.state_dict(), "chart_model.pth")
    return model


def analyze_chart(image_path, model_path="chart_model.pth", image_dir="test_charts/new", json_dir="test_charts/new"):
    """Analyze a single chart image and return detected patterns"""
    # Check if model exists, train if it doesn't
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Training new model...")
        model = train_model(image_dir, json_dir)
    else:
        # Load model
        model = ChartCNN().to(device)
        model.load_state_dict(torch.load(model_path))

    model.eval()

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Enhanced preprocessing
    chart_box = segment_chart_region(image)
    if chart_box is None:
        # If chart region not detected, use whole image
        h, w = image.shape[:2]
        chart_box = (0, 0, w, h)

    chart_line = extract_chart_line(image, chart_box)

    # Make sure chart_line is grayscale for mask_vertical_noise
    if len(chart_line.shape) > 2:
        processed_image = mask_vertical_noise(cv2.cvtColor(chart_line, cv2.COLOR_BGR2GRAY))
    else:
        processed_image = mask_vertical_noise(chart_line)

    # Convert to PIL and apply transforms
    pil_image = Image.fromarray(processed_image)
    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_image = transform_test(pil_image).unsqueeze(0).to(device)

    # Extract prices using CNN
    with torch.no_grad():
        predicted_prices = model(tensor_image)[0].cpu().numpy()

    # Denormalize prices
    predicted_prices = predicted_prices * 17.5 + 47.5

    # Detect patterns
    patterns = detect_technical_patterns(predicted_prices)

    # Visualize results
    visualize_results(image, predicted_prices, patterns, chart_box)

    return predicted_prices, patterns


def visualize_results(image, prices, patterns, chart_box):
    """Visualize the analyzed chart with detected patterns"""
    # Create a copy for visualization
    vis_image = image.copy()

    # Draw the detected chart region
    if chart_box:
        min_x, min_y, max_x, max_y = chart_box
        cv2.rectangle(vis_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Add text for detected patterns
    y_pos = 30
    for pattern, detected in patterns.items():
        if detected:
            cv2.putText(vis_image, f"Detected: {pattern}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30

    # Save the result
    cv2.imwrite("analyzed_chart.png", vis_image)
    print(f"Analysis complete. Result saved as 'analyzed_chart.png'")


# Main execution
if __name__ == "__main__":
    # Example usage
    image_dir = "test_charts/new"
    json_dir = "test_charts/new"

    # Train or load model
    if os.path.exists("chart_model.pth"):
        print("Loading existing model...")
    else:
        print("Training new model...")
        train_model(image_dir, json_dir)

    # Analyze a sample chart
    sample_image = os.path.join(image_dir, os.listdir(image_dir)[0])
    if not sample_image.startswith('marked_') and sample_image.endswith('.png'):
        prices, patterns = analyze_chart(sample_image)
        print("Detected patterns:", patterns)