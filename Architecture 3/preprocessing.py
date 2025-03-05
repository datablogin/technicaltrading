import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import mode


def calculate_gradient_magnitude(image):
    """Calculate gradient magnitude using Sobel operators"""
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(Gx, Gy)
    return cv2.convertScaleAbs(magnitude)


def segment_chart_region(image):
    """Enhanced chart region segmentation using improved Hough transform"""
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection with optimized thresholds
    gradient_magnitude = calculate_gradient_magnitude(blurred)
    high_threshold = np.percentile(gradient_magnitude.ravel(), 95)
    low_threshold = np.percentile(gradient_magnitude.ravel(), 60)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Detect lines using Hough transform with improved parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)

    if lines is None:
        return None

    # Separate horizontal and vertical lines with stricter tolerance
    tolerance = 5  # degrees
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle of line
        if x2 - x1 == 0:  # Vertical line
            vertical_lines.append((x1, y1, x2, y2))
        else:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < tolerance or abs(angle - 180) < tolerance:
                horizontal_lines.append((x1, y1, x2, y2))
            elif abs(angle - 90) < tolerance or abs(angle + 90) < tolerance:
                vertical_lines.append((x1, y1, x2, y2))

    # Extract coordinates for boundary determination
    if not horizontal_lines or not vertical_lines:
        return None

    # Get y-coordinates for horizontal lines
    horizontal_y = [y1 for x1, y1, x2, y2 in horizontal_lines if abs(y1 - y2) < 5]

    # Get x-coordinates for vertical lines
    vertical_x = [x1 for x1, y1, x2, y2 in vertical_lines if abs(x1 - x2) < 5]

    # Find grid lines based on consistent spacing
    if horizontal_y and len(horizontal_y) > 1:
        diffs = np.diff(sorted(horizontal_y))
        if len(diffs) > 0:
            try:
                spacing_y = mode(diffs)[0][0]
                min_y = min(horizontal_y)
                max_y = max(horizontal_y)
            except:
                min_y, max_y = 0, gray.shape[0]
        else:
            min_y, max_y = 0, gray.shape[0]
    else:
        min_y, max_y = 0, gray.shape[0]

    if vertical_x and len(vertical_x) > 1:
        diffs = np.diff(sorted(vertical_x))
        if len(diffs) > 0:
            try:
                spacing_x = mode(diffs)[0][0]
                min_x = min(vertical_x)
                max_x = max(vertical_x)
            except:
                min_x, max_x = 0, gray.shape[1]
        else:
            min_x, max_x = 0, gray.shape[1]
    else:
        min_x, max_x = 0, gray.shape[1]

    return (min_x, min_y, max_x, max_y)


def extract_chart_line(image, chart_box):
    """Extract the chart line from the identified region"""
    if chart_box is None:
        # If chart region not detected, use whole image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return gray

    min_x, min_y, max_x, max_y = chart_box

    # Extract chart region
    chart_region = image[min_y:max_y, min_x:max_x]

    # If image is color
    if len(image.shape) == 3:
        # Convert to grayscale for consistent processing
        gray_region = cv2.cvtColor(chart_region, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to separate chart line from background
        binary = cv2.adaptiveThreshold(
            gray_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Remove grid lines using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return cleaned
    else:
        # For grayscale images
        binary = cv2.adaptiveThreshold(
            chart_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return cleaned


def mask_vertical_noise(image, window_size=3, threshold_factor=3):
    """Mask vertical noise in the image"""
    # Calculate column averages
    col_averages = np.mean(image, axis=0)
    mean_avg = np.mean(col_averages)
    std_avg = np.std(col_averages)
    threshold = mean_avg + threshold_factor * std_avg

    # Identify noisy columns
    noisy_columns = np.where(col_averages > threshold)[0]

    # Apply 1D median filter
    filtered_image = median_filter(image, size=(window_size, 1))

    # Replace only the noisy columns
    result = image.copy()
    result[:, noisy_columns] = filtered_image[:, noisy_columns]

    return result