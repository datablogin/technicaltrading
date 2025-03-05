import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
from scipy.interpolate import make_interp_spline


# Create directories if they don't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Base directory for generated charts
base_dir = "chart_validation"
image_dir = os.path.join(base_dir, "images")
json_dir = os.path.join(base_dir, "labels")

ensure_dir(base_dir)
ensure_dir(image_dir)
ensure_dir(json_dir)


def generate_price_series(length=100, volatility=0.01, seed=None):
    """Generate a random price series with specified volatility"""
    if seed is not None:
        np.random.seed(seed)

    # Start price
    price = 50
    prices = [price]

    # Generate random walk
    for _ in range(length - 1):
        change = np.random.normal(0, volatility)
        price += change * price
        prices.append(price)

    return np.array(prices)


def add_noise(prices, noise_level=0.005):
    """Add noise to the price series"""
    noise = np.random.normal(0, noise_level, len(prices))
    return prices + noise * prices


def smooth_prices(prices, window=5):
    """Smooth the price series using a spline"""
    x = np.arange(len(prices))
    if len(prices) > window:
        spl = make_interp_spline(x, prices, k=3)
        x_smooth = np.linspace(0, len(prices) - 1, len(prices))
        return spl(x_smooth)
    return prices


def generate_double_bottom(length=100, volatility=0.01, bottom_depth=0.15, seed=None):
    """Generate a double bottom pattern"""
    if seed is not None:
        np.random.seed(seed)

    # Generate three segments for the pattern
    first_segment_len = length // 3
    second_segment_len = length // 3
    third_segment_len = length - first_segment_len - second_segment_len

    # First segment: downtrend to first bottom
    first_segment = np.linspace(60, 50 * (1 - bottom_depth), first_segment_len)

    # Second segment: bounce to middle peak
    second_segment = np.linspace(50 * (1 - bottom_depth), 55, second_segment_len)

    # Third segment: drop to second bottom and rally
    third_segment_x = np.arange(third_segment_len)
    third_segment = np.ones(third_segment_len) * 50 * (1 - bottom_depth)
    third_segment += third_segment_x * 0.2  # Add uptrend after second bottom

    # Combine segments
    pattern = np.concatenate([first_segment, second_segment, third_segment])

    # Add noise and smooth
    pattern = add_noise(pattern, volatility)
    pattern = smooth_prices(pattern)

    # Identify key points
    first_bottom_idx = first_segment_len - 1
    middle_peak_idx = first_segment_len + second_segment_len - 1
    second_bottom_idx = min(first_segment_len + second_segment_len + 10, len(pattern) - 10)

    return pattern, [first_bottom_idx, second_bottom_idx], [middle_peak_idx]


def generate_ascending_triangle(length=100, volatility=0.01, breakout_strength=0.2, seed=None):
    """Generate an ascending triangle pattern"""
    if seed is not None:
        np.random.seed(seed)

    # Generate base price
    base_price = 50

    # Generate flat resistance level
    resistance = base_price * 1.1

    # Generate ascending support line
    support_start = base_price * 0.9
    support_end = resistance * 0.95

    # Pre-breakout phase (80% of length)
    pre_breakout_len = int(length * 0.8)

    # Create price series with oscillations between support and resistance
    prices = []
    support_line = []

    for i in range(pre_breakout_len):
        # Support line increases linearly
        support = support_start + (support_end - support_start) * i / pre_breakout_len
        support_line.append(support)

        # Price oscillates between support and resistance
        oscillation_factor = (i / pre_breakout_len) ** 0.5  # Decrease oscillation amplitude over time
        max_oscillation = (resistance - support) * oscillation_factor

        if i % 20 < 10:  # Create oscillation pattern
            prices.append(resistance - max_oscillation * random.uniform(0, 0.5))
        else:
            prices.append(support + max_oscillation * random.uniform(0, 0.5))

    # Breakout phase
    breakout_prices = np.linspace(resistance, resistance * (1 + breakout_strength), length - pre_breakout_len)
    breakout_prices = add_noise(breakout_prices, volatility * 0.5)

    # Combine pre-breakout and breakout phases
    pattern = np.concatenate([prices, breakout_prices])

    # Add noise and smooth
    pattern = add_noise(pattern, volatility)
    pattern = smooth_prices(pattern)

    # Identify key points
    # Find peaks (resistance tests)
    peaks = []
    for i in range(1, pre_breakout_len - 1):
        if pattern[i] > pattern[i - 1] and pattern[i] > pattern[i + 1] and pattern[i] > base_price:
            peaks.append(i)

    # Find troughs (support tests)
    troughs = []
    for i in range(1, pre_breakout_len - 1):
        if pattern[i] < pattern[i - 1] and pattern[i] < pattern[i + 1]:
            troughs.append(i)

    return pattern, troughs, peaks


def generate_head_and_shoulders(length=100, volatility=0.01, seed=None):
    """Generate a head and shoulders pattern"""
    if seed is not None:
        np.random.seed(seed)

    # Divide length into 5 segments
    segment_len = length // 5

    # Generate the pattern
    # Left shoulder
    left_shoulder_peak = 55
    left_shoulder = np.linspace(50, left_shoulder_peak, segment_len)
    left_shoulder = np.concatenate([left_shoulder, np.linspace(left_shoulder_peak, 48, segment_len)])

    # Head
    head_peak = 60
    head = np.linspace(48, head_peak, segment_len)
    head = np.concatenate([head, np.linspace(head_peak, 48, segment_len)])

    # Right shoulder
    right_shoulder_peak = 54
    right_shoulder = np.linspace(48, right_shoulder_peak, segment_len)
    right_shoulder = np.concatenate([right_shoulder, np.linspace(right_shoulder_peak, 46, segment_len // 2)])

    # Combine segments
    pattern = np.concatenate([left_shoulder, head, right_shoulder])

    # Add noise and smooth
    pattern = add_noise(pattern, volatility)
    pattern = smooth_prices(pattern)

    # Identify key points
    left_shoulder_idx = segment_len
    head_idx = 3 * segment_len
    right_shoulder_idx = 5 * segment_len

    # Find neckline points
    left_trough_idx = 2 * segment_len
    right_trough_idx = 4 * segment_len

    return pattern[:length], [left_trough_idx, right_trough_idx], [left_shoulder_idx, head_idx, right_shoulder_idx]


def save_chart(prices, troughs, peaks, pattern_name, index, days=100):
    """Save chart as image and JSON label"""
    # Generate image
    plt.figure(figsize=(10, 6))
    plt.plot(prices, color='black', linewidth=2)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Mark peaks and troughs
    plt.scatter(peaks, prices[peaks], color='green', marker='^', s=100)
    plt.scatter(troughs, prices[troughs], color='red', marker='v', s=100)

    # Add title and labels
    plt.title(f"Oil - {pattern_name} (Buy)", fontsize=16)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Oil Price ($)", fontsize=12)

    # Add some noise chart for background (like volume)
    noise = generate_price_series(days, volatility=0.05, seed=index)
    plt.plot(noise, color='blue', alpha=0.5, linewidth=1)

    # Save image
    image_path = os.path.join(image_dir, f"{pattern_name.lower().replace(' ', '_')}_{index}.png")
    plt.savefig(image_path, dpi=100, bbox_inches='tight')
    plt.close()

    # Save label
    label = {
        "prices": prices.tolist(),
        "troughs": troughs.tolist(),
        "peaks": peaks.tolist(),
        "pattern": pattern_name
    }

    json_path = os.path.join(json_dir, f"{pattern_name.lower().replace(' ', '_')}_{index}.json")
    with open(json_path, 'w') as f:
        json.dump(label, f, indent=2)

    return image_path, json_path


def generate_validation_dataset(num_samples=20):
    """Generate a dataset of charts with different patterns"""
    dataset = []

    # Generate double bottom patterns
    for i in range(num_samples):
        volatility = random.uniform(0.005, 0.02)
        bottom_depth = random.uniform(0.1, 0.2)
        pattern, troughs, peaks = generate_double_bottom(volatility=volatility, bottom_depth=bottom_depth, seed=i)
        img_path, json_path = save_chart(pattern, troughs, peaks, "Double Bottom", i)
        dataset.append({
            "image": img_path,
            "label": json_path,
            "pattern": "Double Bottom"
        })

    # Generate ascending triangle patterns
    for i in range(num_samples):
        volatility = random.uniform(0.005, 0.02)
        breakout_strength = random.uniform(0.15, 0.3)
        pattern, troughs, peaks = generate_ascending_triangle(volatility=volatility,
                                                              breakout_strength=breakout_strength,
                                                              seed=i + 100)
        img_path, json_path = save_chart(pattern, troughs, peaks, "Ascending Triangle", i)
        dataset.append({
            "image": img_path,
            "label": json_path,
            "pattern": "Ascending Triangle"
        })

    # Generate head and shoulders patterns
    for i in range(num_samples):
        volatility = random.uniform(0.005, 0.02)
        pattern, troughs, peaks = generate_head_and_shoulders(volatility=volatility, seed=i + 200)
        img_path, json_path = save_chart(pattern, troughs, peaks, "Head and Shoulders", i)
        dataset.append({
            "image": img_path,
            "label": json_path,
            "pattern": "Head and Shoulders"
        })

    # Save dataset information
    dataset_path = os.path.join(base_dir, "dataset.json")
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} chart images with labels")
    print(f"Dataset information saved to {dataset_path}")

    return dataset


# Generate the validation dataset
if __name__ == "__main__":
    generate_validation_dataset(num_samples=20)