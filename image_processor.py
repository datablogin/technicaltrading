# image_processor.py (updated extract_price_data)
import cv2
import numpy as np

def extract_price_data(image_path):
    # Load the image in color to check for transparency or color issues, then convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to load image")

    # Convert to grayscale, ensuring proper handling of color or transparency
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Preprocess: Enhance contrast minimally, reduce noise lightly, and focus on black line
    gray = cv2.equalizeHist(gray)  # Enhance contrast
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Smallest kernel for minimal noise reduction, preserving line details
    gray = cv2.convertScaleAbs(gray, alpha=1.1, beta=10)  # Slight contrast boost, shift brightness to emphasize black

    # Debug: Print the raw image shape and pixel value distribution
    print(f"Image shape: {height}x{width}")
    print(
        f"Sample pixel values (top row, middle, bottom row): {gray[0, :10]}, {gray[height // 2, :10]}, {gray[-1, :10]}")
    print(f"Pixel value histogram (unique values and counts): {np.unique(gray, return_counts=True)}")

    # Apply adaptive thresholding with adjusted parameters to isolate the chart line (black on white)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 9, -7)  # Black line = 255, white background = 0
    # Apply minimal morphological operations to enhance the chart line, ignoring vertical bars
    kernel = np.ones((3, 3), np.uint8)  # Small kernel for fine control
    thresh = cv2.dilate(thresh, kernel, iterations=1)  # Light enhancement
    thresh = cv2.erode(thresh, kernel, iterations=1)  # Refine to reduce noise
    # Apply vertical structuring element to remove vertical bars (noise)
    vertical_kernel = np.ones((height // 10, 1), np.uint8)  # Tall, thin kernel to remove vertical lines
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)  # Open to remove vertical bars
    print(
        f"Thresholded sample (top row, middle, bottom row): {thresh[0, :10]}, {thresh[height // 2, :10]}, {thresh[-1, :10]}")
    print(f"Threshold pixel histogram (unique values and counts): {np.unique(thresh, return_counts=True)}")

    # Fallback: Apply Canny edge detection with adjusted thresholds, focusing on the chart line
    edges = cv2.Canny(gray, 15, 80)  # Moderate thresholds to detect chart line, reduce vertical bar noise
    print(f"Canny edges sample (top row, middle, bottom row): {edges[0, :10]}, {edges[height // 2, :10]}, {edges[-1, :10]}")
    print(f"Canny edges pixel histogram (unique values and counts): {np.unique(edges, return_counts=True)}")

    # Use probabilistic Hough transform to detect lines, prioritizing edges and ignoring vertical bars
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=3, minLineLength=30, maxLineGap=40)
    print(f"Hough lines detected: {len(lines) if lines is not None else 0}")

    if lines is not None:
        # Filter lines to keep only nearly horizontal lines (slope close to 0), ignoring vertical bars
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                # Focus on very horizontal lines (e.g., |slope| < 0.03), ignoring vertical bars
                if abs(slope) < 0.03 and abs(x2 - x1) > abs(y2 - y1) * 5:  # Ensure line is significantly longer horizontally
                    horizontal_lines.append(line)
        print(f"Filtered horizontal lines detected: {len(horizontal_lines)}")

        if horizontal_lines:
            # Find the most common y-coordinate among horizontal lines (mode)
            y_coords = [(line[0][1] + line[0][3]) / 2 for line in horizontal_lines]  # Average y for each line
            if y_coords:
                # Use the median y-coordinate to represent the chart line
                chart_line_y = np.median(y_coords)
                print(f"Selected chart line y-coordinate: {chart_line_y}")
                # Define a moderate band (e.g., 15 rows centered at the detected line)
                band_start = max(0, int(chart_line_y) - 7)
                band_end = min(height, int(chart_line_y) + 8)
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

    # Clip to ensure values are within 0-255 (thresholded values are 0 or 255)
    prices = np.clip(prices, 0, 255)

    # Debug: Print raw and clipped prices
    print(f"Raw prices (after thresholding): {prices[:10]}...{prices[-10:]}")
    print(f"Clipped prices (0-255): {prices[:10]}...{prices[-10:]}")

    # Check for variation in prices
    print(f"Min price (grayscale): {np.min(prices)}, Max price (grayscale): {np.max(prices)}")

    # Scale prices to approximate chart range ($47.5–$65), ensuring breakout is captured
    min_price, max_price = 47.5, 65.0  # Chart y-axis range from your image
    if np.max(prices) == np.min(prices):  # Check for no variation
        print("Warning: No variation in prices - using default range")
        normalized_prices = np.linspace(min_price, max_price, len(prices))  # Fallback: linear interpolation
    else:
        # Scale to match chart (higher intensity = lower price, invert for black line = high price)
        normalized_prices = min_price + (max_price - min_price) * (1 - (prices - np.min(prices)) / (
            np.max(prices) - np.min(prices)))  # Invert for black line = high price
    print(f"Normalized prices: {normalized_prices[:10]}...{normalized_prices[-10:]}")

    return normalized_prices

# Keep detect_pattern and markup_image as they are
def detect_pattern(prices):
    troughs = []
    peaks = []
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            troughs.append(i)
        elif prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            peaks.append(i)
    return troughs, peaks

def markup_image(image_path, prices, troughs, peaks):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    for i in troughs:
        x = int(i * width / len(prices))
        cv2.line(img, (x, 0), (x, height), (0, 255, 0), 2)  # Green for troughs
    for i in peaks:
        x = int(i * width / len(prices))
        cv2.line(img, (x, 0), (x, height), (0, 0, 255), 2)  # Red for peaks
    return img