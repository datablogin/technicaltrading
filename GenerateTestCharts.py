import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
output_dir = "test_charts"
os.makedirs(output_dir, exist_ok=True)

# Simulate oil price data with noise
def generate_base_data(length=100, base_price=50, volatility=5):
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, volatility, length)
    time = np.arange(length)
    return base_price + noise + time * 0.1  # Slight upward trend + noise

# Generate and save a chart
def save_chart(data, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(data, color='black', linewidth=1.5)
    plt.title(title)
    plt.xlabel("Time (Days)")
    plt.ylabel("Oil Price ($)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# 1. Buy Charts (Double Bottom, Ascending Triangle, Inverse Head and Shoulders)
def generate_buy_charts():
    # Double Bottom
    data = generate_base_data(100, 50, 2)
    data[20:40] = 48  # First trough
    data[60:80] = 48  # Second trough
    data[80:] += 5    # Breakout
    save_chart(data, "Oil - Double Bottom (Buy)", "buy_double_bottom.png")

    # Ascending Triangle
    data = generate_base_data(100, 50, 2)
    for i in range(20, 100, 20):
        data[i:i+10] -= 2  # Higher lows
    data[80:] += 5         # Breakout above flat resistance
    save_chart(data, "Oil - Ascending Triangle (Buy)", "buy_ascending_triangle.png")

    # Inverse Head and Shoulders
    data = generate_base_data(100, 50, 2)
    data[20:30] -= 3  # Left shoulder
    data[50:60] -= 6  # Head
    data[80:90] -= 3  # Right shoulder
    data[90:] += 5    # Breakout
    save_chart(data, "Oil - Inverse Head and Shoulders (Buy)", "buy_inverse_hns.png")

# 2. Hold Charts (No Clear Pattern)
def generate_hold_charts():
    # Flat Trend
    data = generate_base_data(100, 50, 1)
    save_chart(data, "Oil - Flat Trend (Hold)", "hold_flat.png")

    # Random Noise
    data = generate_base_data(100, 50, 3)
    save_chart(data, "Oil - Random Noise (Hold)", "hold_noise.png")

    # Slight Decline
    data = generate_base_data(100, 50, 2) - np.arange(100) * 0.05
    save_chart(data, "Oil - Slight Decline (Hold)", "hold_decline.png")

# 3. Sell Charts (Head and Shoulders, Double Top, Descending Triangle)
def generate_sell_charts():
    # Head and Shoulders
    data = generate_base_data(100, 50, 2)
    data[20:30] += 3  # Left shoulder
    data[50:60] += 6  # Head
    data[80:90] += 3  # Right shoulder
    data[90:] -= 5    # Breakdown
    save_chart(data, "Oil - Head and Shoulders (Sell)", "sell_hns.png")

    # Double Top
    data = generate_base_data(100, 50, 2)
    data[30:40] += 5  # First top
    data[70:80] += 5  # Second top
    data[80:] -= 5    # Breakdown
    save_chart(data, "Oil - Double Top (Sell)", "sell_double_top.png")

    # Descending Triangle
    data = generate_base_data(100, 50, 2)
    for i in range(20, 100, 20):
        data[i:i+10] += 2  # Lower highs
    data[80:] -= 5         # Breakdown below flat support
    save_chart(data, "Oil - Descending Triangle (Sell)", "sell_descending_triangle.png")

# Generate all charts
generate_buy_charts()
generate_hold_charts()
generate_sell_charts()

print(f"Generated 9 test charts in '{output_dir}' folder.")