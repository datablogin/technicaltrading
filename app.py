from flask import Flask, request, jsonify
import cv2
import numpy as np
from scipy.signal import find_peaks
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def extract_price_data(image_path):
    # Load the image in color to check for transparency or color issues, then convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to load image")

    # Convert to grayscale, ensuring proper handling of color or transparency
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Preprocess: Enhance contrast using histogram equalization
    gray = cv2.equalizeHist(gray)

    # Debug: Print the raw image shape and pixel value distribution
    print(f"Image shape: {height}x{width}")
    print(
        f"Sample pixel values (top row, middle, bottom row): {gray[0, :10]}, {gray[height // 2, :10]}, {gray[-1, :10]}")
    print(f"Pixel value histogram (unique values and counts): {np.unique(gray, return_counts=True)}")

    # Apply adaptive thresholding with adjusted parameters to isolate the chart line
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)
    print(
        f"Thresholded sample (top row, middle, bottom row): {thresh[0, :10]}, {thresh[height // 2, :10]}, {thresh[-1, :10]}")
    print(f"Threshold pixel histogram (unique values and counts): {np.unique(thresh, return_counts=True)}")

    # Fallback: Apply Canny edge detection with adjusted thresholds
    edges = cv2.Canny(gray, 50, 150)  # Lower thresholds to capture more edges
    print(f"Canny edges sample (top row, middle, bottom row): {edges[0, :10]}, {edges[height // 2, :10]}, {edges[-1, :10]}")
    print(f"Canny edges pixel histogram (unique values and counts): {np.unique(edges, return_counts=True)}")

    # Use probabilistic Hough transform to detect lines, prioritize edges if thresholding is weak
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=30, maxLineGap=30)
    print(f"Hough lines detected: {len(lines) if lines is not None else 0}")

    if lines is not None:
        # Filter lines to keep only nearly horizontal lines (slope close to 0)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                # Consider lines with slope close to 0 (horizontal, e.g., |slope| < 0.2)
                if abs(slope) < 0.2:
                    horizontal_lines.append(line)
        print(f"Filtered horizontal lines detected: {len(horizontal_lines)}")

        if horizontal_lines:
            # Find the most common y-coordinate among horizontal lines (mode)
            y_coords = [(line[0][1] + line[0][3]) / 2 for line in horizontal_lines]  # Average y for each line
            if y_coords:
                # Use the median y-coordinate to represent the chart line
                chart_line_y = np.median(y_coords)
                print(f"Selected chart line y-coordinate: {chart_line_y}")
                # Define a wider band (e.g., 10 rows centered at the detected line)
                band_start = max(0, int(chart_line_y) - 5)
                band_end = min(height, int(chart_line_y) + 6)
                # Try thresholded image first
                band = thresh[band_start:band_end, :]
                prices = np.mean(band, axis=0)  # Average across rows of the band
                if np.max(prices) == 0:  # If no variation, use Canny edges
                    band_edges = edges[band_start:band_end, :]
                    prices = np.mean(band_edges, axis=0)
                prices = prices[::-1]  # Invert for chart orientation
            else:
                prices = np.mean(thresh, axis=1)[::-1]  # Fallback to row average if no y-coords
        else:
            prices = np.mean(thresh, axis=1)[::-1]  # Fallback to row average if no horizontal lines
    else:
        prices = np.mean(thresh, axis=1)[::-1]  # Fallback to row average if no lines detected

    print(f"Raw prices (after thresholding): {prices[:10]}...{prices[-10:]}")

    # Clip to ensure values are within 0-255 (thresholded values are 0 or 255)
    prices = np.clip(prices, 0, 255)
    print(f"Clipped prices (0-255): {prices[:10]}...{prices[-10:]}")

    # Check for variation in prices
    print(f"Min price (grayscale): {np.min(prices)}, Max price (grayscale): {np.max(prices)}")

    # Scale prices to approximate chart range ($47.5–$65), ensuring breakout is captured
    min_price, max_price = 47.5, 65.0  # Chart y-axis range from your image
    if np.max(prices) == np.min(prices):  # Check for no variation
        print("Warning: No variation in prices - using default range")
        normalized_prices = np.linspace(min_price, max_price, len(prices))  # Fallback: linear interpolation
    else:
        # Scale to match chart (higher intensity = lower price, invert for chart orientation)
        normalized_prices = min_price + (max_price - min_price) * (prices - np.min(prices)) / (
            np.max(prices) - np.min(prices))
    print(f"Normalized prices: {normalized_prices[:10]}...{normalized_prices[-10:]}")

    return normalized_prices


def detect_pattern(prices):
    # Use raw prices (no smoothing) to preserve breakout detail
    smoothed_prices = prices  # No smoothing

    # Adjust distance and height to find closer troughs/peaks, allowing for noise
    troughs, _ = find_peaks(-smoothed_prices, distance=20,
                            height=50.0)  # Troughs near $50 (higher threshold to capture slight variations)
    peaks, _ = find_peaks(smoothed_prices, distance=20,
                          height=62.5)  # Peaks near $62.5–$65 (higher threshold to capture significant peaks)

    # Print statements for debugging
    print("Original prices:", prices)
    print("Troughs found:", troughs)
    print("Peaks found:", peaks)

    # Double Bottom (Buy) - Look for two similar troughs with an upward breakout
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]  # Last two troughs
        if abs(prices[t1] - prices[t2]) < 5.0:  # Relax threshold for price similarity (wider range)
            peak_between = max(prices[t1:t2])  # Peak between troughs
            if t2 < len(prices) - 1 and prices[
                t2 + 1] > peak_between + 15.0:  # Keep breakout threshold for large jump (~$15)
                return "Double Bottom", "Buy", {"troughs": [t1, t2]}

    # Double Top (Sell) - Look for two similar peaks with a downward breakdown
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]  # Last two peaks
        if abs(prices[p1] - prices[p2]) < 5.0:  # Relax threshold for price similarity
            trough_between = min(prices[p1:p2])  # Trough between peaks
            if p2 < len(prices) - 1 and prices[p2 + 1] < trough_between - 15.0:  # Keep breakdown threshold (~$15)
                return "Double Top", "Sell", {"peaks": [p1, p2]}

    return "No Pattern", "Hold", {}


def markup_image(image_path, pattern, pattern_data):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    prices = extract_price_data(image_path)

    # Map price data to image coordinates
    if pattern == "Double Bottom" and "troughs" in pattern_data:
        troughs = pattern_data["troughs"]
        for t in troughs:
            # Scale x-coordinate based on time index (0 to width)
            x = int(t * width / len(prices))
            # Scale y-coordinate based on price (higher prices at top, lower at bottom)
            y = int((prices[t] - min(prices)) / (max(prices) - min(prices)) * (height - 30) + 15)  # Add buffer
            # Draw green circle at trough
            cv2.circle(img, (x, y), 15, (0, 255, 0), 2)  # Green, smaller radius for clarity

    elif pattern == "Double Top" and "peaks" in pattern_data:
        peaks = pattern_data["peaks"]
        for p in peaks:
            # Scale x-coordinate based on time index
            x = int(p * width / len(prices))
            # Scale y-coordinate based on price (higher prices at top, lower at bottom)
            y = int((prices[p] - min(prices)) / (max(prices) - min(prices)) * (height - 30) + 15)  # Add buffer
            # Draw red circle at peak
            cv2.circle(img, (x, y), 15, (0, 0, 255), 2)  # Red, smaller radius for clarity

    # No markup for "No Pattern"
    output_path = "static/output.png"
    cv2.imwrite(output_path, img)
    return output_path


@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image_path = "static/uploaded.png"
    file.save(image_path)

    # Process image
    prices = extract_price_data(image_path)
    pattern, decision, pattern_data = detect_pattern(prices)
    marked_image = markup_image(image_path, pattern, pattern_data)

    # Use full URL for local development
    base_url = "http://localhost:5000"  # Adjust if Flask runs on a different port
    image_url = f"{base_url}/{marked_image}"

    return jsonify({
        "pattern": pattern,
        "decision": decision,
        "image_url": image_url
    })


if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)