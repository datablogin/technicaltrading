import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Ensure output directory exists
output_dir = r"test_charts\new"
os.makedirs(output_dir, exist_ok=True)


# Simulated oil price data with noise
def generate_base_data(length=100, base_price=50, volatility=2):
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, volatility, length)
    time = np.arange(length)
    return base_price + noise + time * 0.1  # Slight upward trend + noise


# Generate and save a chart with annotations, save data as JSON
def save_chart(data, title, filename, troughs=None, peaks=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data, color='black', linewidth=1.5, label="Oil Price")
    plt.title(title)
    plt.xlabel("Time (Days)")
    plt.ylabel("Oil Price ($)")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate troughs (green) and peaks (red) if provided
    if troughs:
        for day in troughs:
            plt.axvline(x=day, color='green', linestyle='--', alpha=0.5)
    if peaks:
        for day in peaks:
            plt.axvline(x=day, color='red', linestyle='--', alpha=0.5)

    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

    # Save chart data as JSON
    chart_data = {
        "title": title,
        "prices": data.tolist(),  # Convert numpy array to list for JSON
        "troughs": troughs if troughs else [],
        "peaks": peaks if peaks else [],
        "pattern": title.split(" (")[1].split(")")[0]  # Extract pattern (e.g., "Buy", "Sell", "Hold")
    }
    json_filename = os.path.splitext(filename)[0] + ".json"
    with open(os.path.join(output_dir, json_filename), 'w') as f:
        json.dump(chart_data, f, indent=4)

    return chart_data


# 1. Buy Charts (Double Bottom, Ascending Triangle, Inverse Head and Shoulders, additional Double Bottoms)
def generate_buy_charts():
    # Double Bottom 1 (Original)
    data = generate_base_data(100, 50, 2)
    data[20:30] = 47.5  # First trough
    data[40:50] = 47.5  # Second trough
    data[60:] += 17.5  # Breakout to 65.0
    save_chart(data, "Oil - Double Bottom (Buy)", "buy_double_bottom.png", troughs=[20, 40], peaks=[80])

    # Double Bottom 2 (Variation)
    data = generate_base_data(100, 50, 2)
    data[15:25] = 47.5  # First trough
    data[45:55] = 47.5  # Second trough
    data[65:] += 17.5  # Breakout to 65.0
    save_chart(data, "Oil - Double Bottom 2 (Buy)", "buy_double_bottom_2.png", troughs=[20, 50], peaks=[70])

    # Ascending Triangle
    data = generate_base_data(100, 50, 2)
    for i in range(20, 100, 20):
        data[i:i + 10] -= 2  # Higher lows
    data[80:] += 15  # Breakout above flat resistance to 65.0
    save_chart(data, "Oil - Ascending Triangle (Buy)", "buy_ascending_triangle.png", troughs=[20, 40, 60], peaks=[80])

    # Inverse Head and Shoulders
    data = generate_base_data(100, 50, 2)
    data[20:30] -= 2.5  # Left shoulder
    data[50:60] -= 5  # Head
    data[80:90] -= 2.5  # Right shoulder
    data[90:] += 17.5  # Breakout to 65.0
    save_chart(data, "Oil - Inverse Head and Shoulders (Buy)", "buy_inverse_hns.png", troughs=[25, 55, 85], peaks=[95])


# 2. Hold Charts (No Clear Pattern, additional variations)
def generate_hold_charts():
    # Flat Trend (Original)
    data = generate_base_data(100, 50, 1)
    save_chart(data, "Oil - Flat Trend (Hold)", "hold_flat.png", troughs=[20, 40, 60, 80, 100])

    # Random Noise
    data = generate_base_data(100, 50, 3)
    save_chart(data, "Oil - Random Noise (Hold)", "hold_noise.png", troughs=[25, 45, 65, 85], peaks=[35, 55, 75, 95])

    # Slight Decline
    data = generate_base_data(100, 50, 2) - np.arange(100) * 0.05
    save_chart(data, "Oil - Slight Decline (Hold)", "hold_decline.png", troughs=[20, 40, 60, 80],
               peaks=[30, 50, 70, 90])

    # Flat Trend 2 (Variation)
    data = generate_base_data(100, 50, 1.5) + np.random.normal(0, 0.5, 100)
    save_chart(data, "Oil - Flat Trend 2 (Hold)", "hold_flat_2.png", troughs=[15, 35, 55, 75, 95],
               peaks=[25, 45, 65, 85])


# 3. Sell Charts (Head and Shoulders, Double Top, Descending Triangle, additional Double Tops)
def generate_sell_charts():
    # Head and Shoulders
    data = generate_base_data(100, 50, 2)
    data[20:30] += 15  # Left shoulder
    data[50:60] += 17.5  # Head
    data[80:90] += 15  # Right shoulder
    data[90:] -= 17.5  # Breakdown to 47.5
    save_chart(data, "Oil - Head and Shoulders (Sell)", "sell_hns.png", peaks=[25, 55, 85], troughs=[95])

    # Double Top 1 (Original)
    data = generate_base_data(100, 50, 2)
    data[30:40] += 15  # First top
    data[70:80] += 15  # Second top
    data[80:] -= 17.5  # Breakdown to 47.5
    save_chart(data, "Oil - Double Top (Sell)", "sell_double_top.png", peaks=[35, 75], troughs=[85])

    # Double Top 2 (Variation)
    data = generate_base_data(100, 50, 2)
    data[25:35] += 15  # First top
    data[65:75] += 15  # Second top
    data[75:] -= 17.5  # Breakdown to 47.5
    save_chart(data, "Oil - Double Top 2 (Sell)", "sell_double_top_2.png", peaks=[30, 70], troughs=[80])

    # Descending Triangle
    data = generate_base_data(100, 50, 2)
    for i in range(20, 100, 20):
        data[i:i + 10] += 2  # Lower highs
    data[80:] -= 15  # Breakdown below flat support to 47.5
    save_chart(data, "Oil - Descending Triangle (Sell)", "sell_descending_triangle.png", peaks=[20, 40, 60],
               troughs=[80])


# Generate all charts
generate_buy_charts()
generate_hold_charts()
generate_sell_charts()

print(f"Generated 15 test charts in '{output_dir}' folder.")