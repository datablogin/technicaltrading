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
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.stats import mode

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============= Preprocessing Functions =============

def safe_mean(arr):
    """Calculate mean safely, handling empty arrays"""
    if len(arr) == 0:
        return 0.0
    return np.mean(arr)

def safe_std(arr):
    """Calculate standard deviation safely, handling empty arrays"""
    if len(arr) < 2:
        return 0.0
    return np.std(arr)

def calculate_gradient_magnitude(image):
    """Calculate gradient magnitude using Sobel operators"""
    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(Gx, Gy)
    return cv2.convertScaleAbs(magnitude)


def segment_chart_region(image):
    """Enhanced chart region segmentation using Hough transform"""
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection with optimized thresholds
    gradient_magnitude = calculate_gradient_magnitude(blurred)
    high_threshold = np.percentile(gradient_magnitude.ravel(), 90)
    low_threshold = np.percentile(gradient_magnitude.ravel(), 50)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)

    if lines is None:
        # If no lines detected, use the whole image
        return None

    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle of line
        if abs(x2 - x1) < 5:  # Nearly vertical line
            vertical_lines.append((x1, y1, x2, y2))
        elif abs(y2 - y1) < 5:  # Nearly horizontal line
            horizontal_lines.append((x1, y1, x2, y2))

    # If not enough lines, use the whole image
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    # Extract coordinates for boundaries
    horizontal_y = [y1 for x1, y1, x2, y2 in horizontal_lines]
    vertical_x = [x1 for x1, y1, x2, y2 in vertical_lines]

    # Set chart boundaries
    min_x = min(vertical_x)
    max_x = max(vertical_x)
    min_y = min(horizontal_y)
    max_y = max(horizontal_y)

    return (min_x, min_y, max_x, max_y)


def extract_chart_line(image, chart_box):
    """Extract the chart line from the identified region"""
    if chart_box is None:
        # If chart region not detected, use whole image
        h, w = image.shape[:2]
        chart_box = (0, 0, w, h)

    min_x, min_y, max_x, max_y = chart_box

    # Extract chart region
    chart_region = image[min_y:max_y, min_x:max_x]

    # If image is color
    if len(image.shape) == 3:
        # Convert to grayscale
        gray_region = cv2.cvtColor(chart_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_region = chart_region

    return gray_region


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


# ============= Pattern Detection Functions =============

def detect_peaks_and_troughs(prices, prominence=0.05, distance=5):
    """Detect peaks and troughs in price data"""
    # Normalize prices for consistent detection
    min_price = np.min(prices)
    max_price = np.max(prices)
    if max_price == min_price:  # Avoid division by zero
        norm_prices = prices - min_price
    else:
        norm_prices = (prices - min_price) / (max_price - min_price)

    # Find peaks
    peaks, peak_props = find_peaks(norm_prices, prominence=prominence, distance=distance)

    # Find troughs by inverting
    troughs, trough_props = find_peaks(-norm_prices, prominence=prominence, distance=distance)

    return peaks, troughs, peak_props, trough_props


def detect_double_bottom(prices, peaks, troughs):
    """Detect double bottom pattern (W-shape)"""
    if len(troughs) < 2 or len(peaks) < 1:
        return False

    # Find all potential double bottoms
    for i in range(len(troughs) - 1):
        if i + 1 >= len(troughs):
            continue

        t1, t2 = troughs[i], troughs[i + 1]

        # Ensure there's enough space between troughs
        if t2 - t1 < 3:
            continue

        # Find peak between the two troughs
        middle_peaks = [p for p in peaks if t1 < p < t2]

        if middle_peaks:
            middle_peak = middle_peaks[0]

            # Check if troughs are at similar levels
            trough1_val = prices[t1]
            trough2_val = prices[t2]
            peak_val = prices[middle_peak]

            trough_diff = abs(trough1_val - trough2_val)
            trough_avg = (trough1_val + trough2_val) / 2

            # More lenient criteria: Troughs should be roughly at similar levels
            # and peak should be higher
            if trough_diff / (trough_avg + 0.0001) < 0.2 and peak_val > trough_avg * 1.05:
                return True

    return False


def detect_head_and_shoulders(prices, peaks, troughs):
    """Detect head and shoulders pattern"""
    if len(peaks) < 3:
        return False

    # Need at least 3 consecutive peaks
    for i in range(len(peaks) - 2):
        if i + 2 >= len(peaks):
            continue

        p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]

        # Check for minimum distance between peaks
        if p2 - p1 < 2 or p3 - p2 < 2:
            continue

        # Check if middle peak (head) is higher than shoulders
        if prices[p2] > prices[p1] and prices[p2] > prices[p3]:
            # Check if shoulders are at similar levels (more lenient)
            shoulder_diff = abs(prices[p1] - prices[p3])
            shoulder_avg = (prices[p1] + prices[p3]) / 2

            if shoulder_diff / (shoulder_avg + 0.0001) < 0.2:
                return True

    return False


def detect_triangle(prices, peaks, troughs):
    """Detect triangle patterns (ascending, descending, symmetric)"""
    if len(peaks) < 2 or len(troughs) < 2:
        return None

    # Check if we have enough data points for trend analysis
    if len(peaks) >= 2 and len(troughs) >= 2:
        # Extract peak and trough values
        peak_indices = np.array(peaks)
        peak_values = prices[peak_indices]

        trough_indices = np.array(troughs)
        trough_values = prices[trough_indices]

        # Check for trends in highs and lows if we have enough points
        if len(peak_values) >= 2:
            x_peaks = np.arange(len(peak_values))
            peak_trend = np.polyfit(x_peaks, peak_values, 1)[0]
        else:
            peak_trend = 0

        if len(trough_values) >= 2:
            x_troughs = np.arange(len(trough_values))
            trough_trend = np.polyfit(x_troughs, trough_values, 1)[0]
        else:
            trough_trend = 0

        # More lenient criteria for triangle detection
        if peak_trend < -0.005 and trough_trend > 0.005:
            return "Symmetric Triangle"
        elif peak_trend < -0.005 and abs(trough_trend) < 0.01:
            return "Descending Triangle"
        elif abs(peak_trend) < 0.01 and trough_trend > 0.005:
            return "Ascending Triangle"

    return None


def detect_trend(prices):
    """Detect simple trend patterns"""
    if len(prices) < 5:
        return None

    # Calculate trend using linear regression
    x = np.arange(len(prices))
    slope = np.polyfit(x, prices, 1)[0]

    # Determine trend strength
    if slope > 0.1:
        return "Strong Uptrend"
    elif slope > 0.02:
        return "Uptrend"
    elif slope < -0.1:
        return "Strong Downtrend"
    elif slope < -0.02:
        return "Downtrend"
    else:
        return "Sideways"


def detect_technical_patterns(prices):
    """Balanced pattern detection with adjusted thresholds"""
    results = {}
    confidence_scores = {}

    # Store all detected patterns with confidence scores
    for window in [3, 5, 8]:  # Multiple window sizes for different scales
        prominence = 0.02 if window == 3 else (0.04 if window == 5 else 0.06)

        # Get peaks and troughs
        peaks, troughs, _, _ = detect_peaks_and_troughs(prices, prominence=prominence, distance=window)

        print(f"Window size {window}: Found {len(peaks)} peaks and {len(troughs)} troughs")

        # Check patterns with enhanced detectors
        db_conf = check_double_bottom_confidence(prices, peaks, troughs)
        if db_conf > 0:
            confidence_scores['Double Bottom'] = max(confidence_scores.get('Double Bottom', 0), db_conf)

        hs_conf = check_head_shoulders_confidence(prices, peaks, troughs)
        if hs_conf > 0:
            confidence_scores['Head and Shoulders'] = max(confidence_scores.get('Head and Shoulders', 0), hs_conf)

        triangle_type, triangle_conf = check_triangle_confidence(prices, peaks, troughs)
        if triangle_type and triangle_conf > 0:
            confidence_scores[triangle_type] = max(confidence_scores.get(triangle_type, 0), triangle_conf)

    # Add trend detection
    trend, trend_conf = check_trend_confidence(prices)
    confidence_scores[f'Trend: {trend}'] = trend_conf

    # Use pattern-specific thresholds with bias correction
    # Make double bottom detection harder, ascending triangle detection easier
    thresholds = {
        'Double Bottom': 0.78,  # Higher threshold to reduce false positives
        'Head and Shoulders': 0.60,  # Medium threshold
        'Ascending Triangle': 0.45,  # Lower threshold to increase detection
        'Descending Triangle': 0.60,
        'Symmetric Triangle': 0.60
    }

    # If multiple patterns exceed thresholds, prefer certain patterns based on confidence ratios
    pattern_priorities = {
        'Ascending Triangle': 1.2,  # Boost ascending triangle confidence
        'Head and Shoulders': 1.1,  # Slightly boost head and shoulders
        'Double Bottom': 0.9  # Slightly reduce double bottom
    }

    # Apply bias correction to confidence scores
    adjusted_scores = {}
    for pattern, conf in confidence_scores.items():
        if "Trend" in pattern:
            adjusted_scores[pattern] = conf
        elif pattern in pattern_priorities:
            adjusted_scores[pattern] = conf * pattern_priorities[pattern]
        else:
            adjusted_scores[pattern] = conf

    # Select patterns meeting thresholds
    for pattern, conf in adjusted_scores.items():
        if "Trend" in pattern:
            # Always include trend
            results[pattern] = conf
        elif pattern in thresholds and conf >= thresholds[pattern]:
            results[pattern] = conf

    # If we have overlapping patterns, use the highest adjusted confidence
    pattern_groups = [
        ['Double Bottom', 'Head and Shoulders'],
        ['Ascending Triangle', 'Descending Triangle', 'Symmetric Triangle']
    ]

    for group in pattern_groups:
        patterns_in_group = [p for p in group if p in results]
        if len(patterns_in_group) > 1:
            # Keep only the highest confidence pattern in the group
            best_pattern = max(patterns_in_group, key=lambda p: adjusted_scores[p])
            for p in patterns_in_group:
                if p != best_pattern:
                    results.pop(p, None)

    return results


def simpler_pattern_detection(prices):
    """Simplified pattern detection focusing on core pattern characteristics"""
    # Calculate basic metrics
    price_range = np.max(prices) - np.min(prices)

    # Get basic peaks and troughs with consistent parameters
    peaks, _ = find_peaks(prices, prominence=0.03 * price_range, distance=3)
    troughs, _ = find_peaks(-prices, prominence=0.03 * price_range, distance=3)

    # Calculate peak and trough heights relative to price range
    peak_heights = prices[peaks] / price_range
    trough_depths = (np.max(prices) - prices[troughs]) / price_range

    # 1. Double Bottom Detection (two similar lows with peak between)
    double_bottom_score = 0.0
    if len(troughs) >= 2:
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                t1, t2 = troughs[i], troughs[j]

                # Skip if troughs are too close
                if t2 - t1 < 10:
                    continue

                # Check if troughs are at similar levels
                trough1_val = prices[t1]
                trough2_val = prices[t2]
                bottom_diff = abs(trough1_val - trough2_val) / price_range

                # If bottoms are similar (within 5% of price range)
                if bottom_diff < 0.05:
                    # Look for peak between the troughs
                    peaks_between = [p for p in peaks if t1 < p < t2]
                    if peaks_between:
                        middle_peak = peaks_between[np.argmax(prices[peaks_between])]
                        peak_height = (prices[middle_peak] - min(trough1_val, trough2_val)) / price_range

                        # If peak is high enough to be significant
                        if peak_height > 0.15:
                            score = (1.0 - bottom_diff / 0.05) * 0.6 + min(1.0, peak_height / 0.3) * 0.4
                            double_bottom_score = max(double_bottom_score, score)

    # 2. Head and Shoulders (three peaks, middle higher)
    head_shoulders_score = 0.0
    if len(peaks) >= 3:
        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]

            # Skip if peaks are too close
            if p2 - p1 < 5 or p3 - p2 < 5:
                continue

            # Get peak heights
            left_shoulder = prices[p1]
            head = prices[p2]
            right_shoulder = prices[p3]

            # Check if middle peak is highest
            if head > left_shoulder and head > right_shoulder:
                # Check if shoulders are at similar heights
                shoulder_diff = abs(left_shoulder - right_shoulder) / price_range

                # If shoulders are similar (within 10% of price range)
                if shoulder_diff < 0.1:
                    # Check if head is significantly higher
                    head_prominence = (head - max(left_shoulder, right_shoulder)) / price_range

                    if head_prominence > 0.1:
                        score = (1.0 - shoulder_diff / 0.1) * 0.6 + min(1.0, head_prominence / 0.2) * 0.4
                        head_shoulders_score = max(head_shoulders_score, score)

    # 3. Ascending Triangle (flat top, rising bottoms)
    ascending_triangle_score = 0.0
    if len(peaks) >= 2 and len(troughs) >= 2:
        # For a simple approach, check if:
        # 1. Top peaks are flat (similar heights)
        top_peaks = sorted(peaks, key=lambda p: prices[p], reverse=True)[:3]
        if len(top_peaks) >= 2:
            top_values = prices[top_peaks]
            top_diff = (np.max(top_values) - np.min(top_values)) / price_range

            # If tops are relatively flat (within 10% of price range)
            if top_diff < 0.1:
                # 2. Check if troughs are rising
                sorted_troughs = sorted(troughs)
                if len(sorted_troughs) >= 3:
                    trough_values = prices[sorted_troughs]

                    # Calculate simple linear trend of trough values
                    trough_indices = np.arange(len(trough_values))
                    trough_slope, _ = np.polyfit(trough_indices, trough_values, 1)

                    # If troughs have positive slope
                    if trough_slope > 0:
                        score = (1.0 - top_diff / 0.1) * 0.5 + min(1.0, trough_slope * 10) * 0.5
                        ascending_triangle_score = max(ascending_triangle_score, score)

    # Determine the most confident pattern
    scores = {
        'Double Bottom': double_bottom_score,
        'Head and Shoulders': head_shoulders_score,
        'Ascending Triangle': ascending_triangle_score
    }

    # Apply threshold and balance patterns
    balanced_scores = {
        # Slightly adjust to prevent double bottom bias
        'Double Bottom': double_bottom_score * 0.9,
        'Head and Shoulders': head_shoulders_score * 1.1,
        'Ascending Triangle': ascending_triangle_score * 1.15  # Boost ascending triangle
    }

    # Select patterns meeting minimum threshold
    results = {}
    min_threshold = 0.55
    for pattern, score in balanced_scores.items():
        if score >= min_threshold:
            results[pattern] = score

    # Add trend info
    trend, trend_conf = check_trend_confidence(prices)
    results[f'Trend: {trend}'] = trend_conf

    return results

# Example confidence check function
def check_double_bottom_confidence(prices, peaks, troughs):
    """More stringent Double Bottom detector to reduce false positives"""
    if len(troughs) < 2 or len(peaks) < 1:
        return 0.0

    max_confidence = 0.0
    sorted_troughs = np.array(sorted(troughs))

    # Try all possible pairs of troughs
    for i in range(len(sorted_troughs) - 1):
        for j in range(i + 1, len(sorted_troughs)):
            t1, t2 = sorted_troughs[i], sorted_troughs[j]

            # Troughs should be reasonably far apart
            if t2 - t1 < 10 or t2 - t1 > len(prices) // 2:
                continue

            # Get values at these troughs
            trough1_val = prices[t1]
            trough2_val = prices[t2]

            # Check if troughs are at similar price levels (critical for double bottom)
            trough_diff = abs(trough1_val - trough2_val)
            trough_avg = (trough1_val + trough2_val) / 2
            bottom_similarity = 1.0 - min(1.0, trough_diff / (trough_avg + 0.0001) / 0.05)

            # If bottoms aren't similar, this isn't a double bottom
            if bottom_similarity < 0.7:
                continue

            # Find middle peak between troughs
            peaks_between = [p for p in peaks if t1 < p < t2]
            if not peaks_between:
                continue

            # Use highest peak between troughs
            middle_peak = peaks_between[np.argmax(prices[peaks_between])]
            peak_val = prices[middle_peak]

            # Middle peak should be significantly higher than troughs
            peak_height = (peak_val - trough_avg) / trough_avg
            peak_significance = min(1.0, peak_height / 0.1)

            if peak_significance < 0.5:  # Peak must be significantly higher
                continue

            # Middle peak should be roughly centered
            position_ratio = (middle_peak - t1) / (t2 - t1)
            position_score = 1.0 - min(1.0, abs(0.5 - position_ratio) / 0.2)

            # Check for upward movement after the second trough
            breakout_score = 0.0
            if t2 < len(prices) - 5:
                post_bottom = prices[t2:]
                if len(post_bottom) > 0 and np.max(post_bottom) > peak_val:
                    breakout_score = 0.3

            # Calculate overall confidence with strict criteria
            confidence = (
                    bottom_similarity * 0.4 +
                    peak_significance * 0.3 +
                    position_score * 0.2 +
                    breakout_score * 0.1
            )

            # Must have VERY similar bottoms and good peak height
            if bottom_similarity > 0.9 and peak_significance > 0.7:
                confidence = min(1.0, confidence * 1.1)

            max_confidence = max(max_confidence, confidence)

    return max_confidence


def check_head_shoulders_confidence(prices, peaks, troughs):
    """Improved Head and Shoulders detector with specific pattern features"""
    if len(peaks) < 3:
        return 0.0

    max_confidence = 0.0

    # Find sequences of 3 consecutive peaks that could form H&S
    for i in range(len(peaks) - 2):
        p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]

        # Ensure proper spacing between peaks
        if p2 - p1 < 5 or p3 - p2 < 5:  # Need sufficient space between peaks
            continue

        # Get peak values
        left_shoulder = prices[p1]
        head = prices[p2]
        right_shoulder = prices[p3]

        # Check core H&S criteria: middle peak must be highest
        if head > left_shoulder and head > right_shoulder:
            # Calculate shoulder symmetry (should be similar heights)
            shoulder_diff = abs(left_shoulder - right_shoulder)
            shoulder_avg = (left_shoulder + right_shoulder) / 2
            shoulder_symmetry = 1.0 - min(1.0, shoulder_diff / (shoulder_avg + 0.0001) / 0.15)

            # Check head prominence (head should be significantly higher than shoulders)
            head_height = (head - shoulder_avg) / shoulder_avg
            head_prominence = min(1.0, head_height / 0.15)

            # Find troughs between peaks for neckline
            left_trough_idx = None
            right_trough_idx = None

            # Find single troughs between peaks
            for t in troughs:
                if p1 < t < p2 and (left_trough_idx is None or prices[t] < prices[left_trough_idx]):
                    left_trough_idx = t

            for t in troughs:
                if p2 < t < p3 and (right_trough_idx is None or prices[t] < prices[right_trough_idx]):
                    right_trough_idx = t

            # Check for neckline (should be relatively flat)
            neckline_score = 0.0
            if left_trough_idx is not None and right_trough_idx is not None:
                neckline_diff = abs(prices[left_trough_idx] - prices[right_trough_idx])
                neckline_avg = (prices[left_trough_idx] + prices[right_trough_idx]) / 2
                neckline_score = 1.0 - min(1.0, neckline_diff / (neckline_avg + 0.0001) / 0.1)

            # Check for downward breakout after pattern
            breakout_score = 0.0
            if p3 < len(prices) - 5:
                post_pattern = prices[p3:min(p3 + 10, len(prices))]

                if left_trough_idx is not None and right_trough_idx is not None:
                    # Use the higher of the two troughs as neckline level
                    neckline_level = max(prices[left_trough_idx], prices[right_trough_idx])

                    # Check if price drops below neckline after pattern
                    if min(post_pattern) < neckline_level:
                        breakout_score = 0.3  # Bonus for having a breakout

            # Calculate overall confidence
            confidence = (
                    shoulder_symmetry * 0.3 +
                    head_prominence * 0.3 +
                    neckline_score * 0.3 +
                    breakout_score * 0.1
            )

            # Apply stricter criteria - must have good symmetry AND prominence
            if shoulder_symmetry > 0.7 and head_prominence > 0.7:
                confidence *= 1.2  # Boost confidence
                confidence = min(1.0, confidence)  # Cap at 1.0

            max_confidence = max(max_confidence, confidence)

    return max_confidence


def check_triangle_confidence(prices, peaks, troughs):
    """Enhanced triangle detector specifically calibrated for ascending triangles"""
    if len(peaks) < 2 or len(troughs) < 2:
        return None, 0.0

    # Check specifically for ascending triangle characteristics

    # 1. Check for breakout at the end of the pattern (critical for ascending triangles)
    breakout_detected = False
    breakout_score = 0.0

    if len(prices) > 20:
        # Look at the last 20% vs previous 20% of data
        breakout_idx = int(len(prices) * 0.8)
        pre_breakout = prices[breakout_idx - 10:breakout_idx]
        post_breakout = prices[breakout_idx:]

        if len(pre_breakout) > 0 and len(post_breakout) > 0:
            pre_avg = np.mean(pre_breakout)
            post_avg = np.mean(post_breakout)

            # Calculate percentage increase
            if pre_avg > 0:  # Prevent division by zero
                pct_increase = (post_avg - pre_avg) / pre_avg

                # Significant upward breakout is critical for ascending triangle
                if pct_increase > 0.05:  # Lower threshold to 5%
                    breakout_detected = True
                    breakout_score = min(1.0, pct_increase / 0.15)  # Higher sensitivity

    # 2. Check for multiple tests of resistance level (flat top)
    if len(peaks) >= 2:
        # Use highest peaks to define resistance
        peak_values = prices[peaks]
        top_peaks = peaks[np.argsort(peak_values)[-3:]]  # Get indices of 3 highest peaks

        if len(top_peaks) >= 2:
            top_values = prices[top_peaks]
            top_std = np.std(top_values)
            top_mean = np.mean(top_values)

            # Check if tops are at similar levels (flat resistance)
            if top_mean > 0:
                resistance_flatness = 1.0 - min(1.0, top_std / top_mean / 0.05)
            else:
                resistance_flatness = 0.0
        else:
            resistance_flatness = 0.0
    else:
        resistance_flatness = 0.0

    # 3. Check for rising support (higher lows)
    if len(troughs) >= 2:
        # Sort troughs by time
        sorted_troughs = sorted(troughs)
        trough_values = prices[sorted_troughs]

        # Check if later troughs are higher than earlier ones
        if len(trough_values) >= 2:
            # Calculate if there's an upward trend in troughs
            trough_diff = []
            for i in range(1, len(trough_values)):
                trough_diff.append(trough_values[i] - trough_values[i - 1])

            # Rising bottoms have positive differences
            positive_diffs = sum(1 for d in trough_diff if d > 0)
            if positive_diffs / max(1, len(trough_diff)) > 0.5:  # More than half are rising
                rising_support = True
                rising_support_score = positive_diffs / max(1, len(trough_diff))
            else:
                rising_support = False
                rising_support_score = 0.0
        else:
            rising_support = False
            rising_support_score = 0.0
    else:
        rising_support = False
        rising_support_score = 0.0

    # Calculate overall confidence for ascending triangle
    if (resistance_flatness > 0.6 and rising_support) or (resistance_flatness > 0.5 and breakout_detected):
        triangle_confidence = (
                resistance_flatness * 0.4 +
                rising_support_score * 0.3 +
                breakout_score * 0.3
        )

        # Boost confidence if all three criteria are met
        if resistance_flatness > 0.6 and rising_support and breakout_detected:
            triangle_confidence *= 1.2
            triangle_confidence = min(1.0, triangle_confidence)  # Cap at 1.0

        if triangle_confidence > 0.5:  # Lower threshold from 0.6 to 0.5
            return "Ascending Triangle", triangle_confidence

    # If not an ascending triangle, return none
    return None, 0.0


def check_trend_confidence(prices):
    """Detect trend and calculate confidence score"""
    if len(prices) < 5:
        return "Unknown", 0.0

    # Calculate trend using linear regression
    x = np.arange(len(prices))
    coeffs = np.polyfit(x, prices, 1)
    slope = coeffs[0]

    # Calculate normalized slope (percentage change per data point)
    normalized_slope = slope / np.mean(prices)

    # Calculate fit quality (how well the points follow the trend)
    trend_line = np.polyval(coeffs, x)
    residuals = np.sqrt(np.mean((prices - trend_line) ** 2))
    fit_quality = 1.0 - min(1.0, residuals / np.mean(prices) / 0.05)

    # Determine trend strength and confidence
    if normalized_slope > 0.005:  # Strong uptrend
        trend_strength = min(1.0, normalized_slope / 0.01)
        confidence = 0.5 + (trend_strength * 0.3 + fit_quality * 0.2)
        return "Strong Uptrend", confidence
    elif normalized_slope > 0.001:  # Uptrend
        trend_strength = min(1.0, normalized_slope / 0.005)
        confidence = 0.5 + (trend_strength * 0.3 + fit_quality * 0.2)
        return "Uptrend", confidence
    elif normalized_slope < -0.005:  # Strong downtrend
        trend_strength = min(1.0, abs(normalized_slope) / 0.01)
        confidence = 0.5 + (trend_strength * 0.3 + fit_quality * 0.2)
        return "Strong Downtrend", confidence
    elif normalized_slope < -0.001:  # Downtrend
        trend_strength = min(1.0, abs(normalized_slope) / 0.005)
        confidence = 0.5 + (trend_strength * 0.3 + fit_quality * 0.2)
        return "Downtrend", confidence
    else:  # Sideways
        trend_strength = 1.0 - min(1.0, abs(normalized_slope) / 0.001)
        confidence = 0.5 + (trend_strength * 0.3 + fit_quality * 0.2)
        return "Sideways", confidence


# ============= Dataset and Model Classes =============

class ChartDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None, window_size=3, threshold_factor=3):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.startswith('marked_')]
        # Load JSON files only if they exist
        self.data = {}
        for img in self.images:
            json_file = os.path.join(json_dir, img.replace('.png', '.json'))
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    self.data[img.replace('.png', '.json')] = json.load(f)
            else:
                # Create dummy data
                self.data[img.replace('.png', '.json')] = {
                    "prices": [50] * 100,  # Dummy prices
                    "troughs": [20, 60],  # Dummy troughs
                    "peaks": [40, 80],  # Dummy peaks
                    "pattern": "Unknown"  # Dummy pattern
                }

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
        chart_box = segment_chart_region(image)
        chart_line = extract_chart_line(image, chart_box)

        # Convert to grayscale if needed and apply noise masking
        if len(chart_line.shape) > 2:
            chart_line = cv2.cvtColor(chart_line, cv2.COLOR_BGR2GRAY)

        chart_line = mask_vertical_noise(chart_line, self.window_size, self.threshold_factor)

        # Convert to PIL for transforms
        chart_line = Image.fromarray(chart_line)

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


# ============= Main Functions =============

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0).to(device)
    prices = torch.stack([label["prices"] for label in labels], 0).to(device)
    troughs = [label["troughs"] for label in labels]
    peaks = [label["peaks"] for label in labels]
    patterns = [label["pattern"] for label in labels]
    return images, {"prices": prices, "troughs": troughs, "peaks": peaks, "pattern": patterns}


def train_model(image_dir, json_dir, num_epochs=5):  # Reduced epochs for testing
    """Train the CNN model"""
    # Data transforms
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

    # Create dataset and dataloader
    dataset = ChartDataset(image_dir=image_dir, json_dir=json_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = ChartCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            prices = labels["prices"]
            loss = criterion(outputs, prices)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
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

    # Make sure chart_line is grayscale
    if len(chart_line.shape) > 2:
        chart_line = cv2.cvtColor(chart_line, cv2.COLOR_BGR2GRAY)

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
    output_filename = f"analyzed_{os.path.basename(image_path)}"
    visualize_results(image, predicted_prices, patterns, chart_box, output_filename)

    # After detecting patterns
    patterns = detect_technical_patterns(predicted_prices)

    # Print trading recommendations
    print("\n===== TRADING RECOMMENDATIONS =====")
    for pattern, confidence in patterns.items():
        if "Trend" not in pattern:  # Skip trend info for recommendations
            pattern_info = get_pattern_description(pattern)
            print(f"\n{pattern} (Confidence: {confidence:.2f})")
            print(f"Description: {pattern_info['description']}")
            print(f"Signal: {pattern_info['signal']}")
            print(f"Target: {pattern_info['target']}")

    # Visualize results
    output_filename = f"analyzed_{os.path.basename(image_path)}"
    visualize_results(image, predicted_prices, patterns, chart_box, output_filename)

    return predicted_prices, patterns


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=8, gap_length=8):
    """
    Draw a dashed line on an image

    Parameters:
    - img: Image to draw on
    - pt1: First point (x1, y1)
    - pt2: Second point (x2, y2)
    - color: Line color (B, G, R)
    - thickness: Line thickness
    - dash_length: Length of each dash
    - gap_length: Length of gaps between dashes
    """
    dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    if dist == 0:
        return img

    dashes = int(dist / (dash_length + gap_length))
    if dashes == 0:
        cv2.line(img, pt1, pt2, color, thickness)
        return img

    unit_x = (pt2[0] - pt1[0]) / dist
    unit_y = (pt2[1] - pt1[1]) / dist

    x1, y1 = pt1
    for i in range(dashes):
        # Draw the dash
        x2 = int(x1 + dash_length * unit_x)
        y2 = int(y1 + dash_length * unit_y)
        cv2.line(img, (int(x1), int(y1)), (x2, y2), color, thickness)

        # Move to the start of the next dash
        x1 = x1 + (dash_length + gap_length) * unit_x
        y1 = y1 + (dash_length + gap_length) * unit_y

    return img


# Then, modify the visualize_results function where the LINE_DASH error occurs
# Replace the line:
# cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_DASH)
# With:
# draw_dashed_line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

def visualize_results(image, prices, patterns, chart_box, output_filename="analyzed_chart.png"):
    """Enhanced visualization with pattern-specific markers and trend lines"""
    vis_image = image.copy()

    # Draw chart region
    if chart_box:
        min_x, min_y, max_x, max_y = chart_box
        cv2.rectangle(vis_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Add pattern labels with confidence scores
    y_pos = 30
    for pattern, confidence in patterns.items():
        # Format confidence as float for cleaner display
        if isinstance(confidence, np.float64):
            confidence = float(confidence)

        # Use different colors for different pattern types
        if "Trend" in pattern:
            color = (255, 165, 0)  # Orange for trends
        else:
            color = (0, 0, 255)  # Red for patterns

        cv2.putText(vis_image, f"{pattern}: {confidence:.2f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_pos += 30

    # Draw pattern-specific visualizations
    if chart_box and len(prices) > 0:
        min_x, min_y, max_x, max_y = chart_box
        width = max_x - min_x
        height = max_y - min_y

        # Scale the prices to match the chart region
        def px(idx):
            return min_x + int(idx / len(prices) * width)

        def py(price):
            # Map price to y coordinate (inverted as y grows downward in image)
            normalized = (price - 47.5) / 17.5  # Same normalization used in your code
            return int(min_y + (max_y - min_y) * (1 - normalized))

        # Draw the price chart line
        for i in range(len(prices) - 1):
            x1 = px(i)
            x2 = px(i + 1)
            y1 = py(prices[i])
            y2 = py(prices[i + 1])
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw pattern-specific markers
        for pattern_name, _ in patterns.items():
            if "Ascending Triangle" in pattern_name:
                # Find resistance level (flat top)
                top_prices = []
                for i in range(len(prices)):
                    if i > 0 and i < len(prices) - 1:
                        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                            top_prices.append(prices[i])

                # Find support level (rising bottom)
                bottom_prices = []
                bottom_indices = []
                for i in range(len(prices)):
                    if i > 0 and i < len(prices) - 1:
                        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                            bottom_prices.append(prices[i])
                            bottom_indices.append(i)

                if top_prices and len(top_prices) >= 2:
                    # Draw resistance line
                    resistance_level = np.mean(top_prices[-3:])  # Use last few tops
                    cv2.line(vis_image,
                             (min_x + int(width * 0.3), py(resistance_level)),
                             (max_x, py(resistance_level)),
                             (0, 255, 255), 2)

                if bottom_indices and len(bottom_indices) >= 2:
                    # Draw support line (rising)
                    # Use linear regression to find the trend line
                    x_points = np.array(bottom_indices[-3:])
                    y_points = np.array(bottom_prices[-3:])

                    if len(x_points) >= 2:
                        # Calculate trend line
                        slope, intercept = np.polyfit(x_points, y_points, 1)

                        # Draw support line
                        start_idx = max(0, bottom_indices[0] - 5)
                        end_idx = min(len(prices) - 1, bottom_indices[-1] + 10)

                        start_price = slope * start_idx + intercept
                        end_price = slope * end_idx + intercept

                        cv2.line(vis_image,
                                 (px(start_idx), py(start_price)),
                                 (px(end_idx), py(end_price)),
                                 (0, 255, 255), 2)

                # Draw breakout point
                if len(prices) > 20:
                    breakout_idx = int(len(prices) * 0.8)
                    cv2.circle(vis_image,
                               (px(breakout_idx), py(prices[breakout_idx])),
                               5, (0, 255, 0), -1)

                    # Draw breakout arrow
                    cv2.arrowedLine(vis_image,
                                    (px(breakout_idx), py(prices[breakout_idx]) - 20),
                                    (px(breakout_idx), py(prices[breakout_idx]) - 5),
                                    (0, 255, 0), 2, tipLength=0.3)

            elif "Double Bottom" in pattern_name:
                # Find the two lowest points and the middle peak
                if len(prices) > 10:
                    # Split into three sections for pattern detection
                    section_size = len(prices) // 3

                    # Find lowest points in first and last sections
                    first_section = prices[:section_size]
                    last_section = prices[-section_size:]

                    if len(first_section) > 0 and len(last_section) > 0:
                        first_bottom_idx = np.argmin(first_section)
                        last_bottom_idx = np.argmin(last_section) + 2 * section_size

                        # Find peak in middle section
                        middle_section = prices[section_size:2 * section_size]
                        if len(middle_section) > 0:
                            middle_peak_idx = np.argmax(middle_section) + section_size

                            # Draw the pattern markers
                            # First bottom
                            cv2.circle(vis_image,
                                       (px(first_bottom_idx), py(prices[first_bottom_idx])),
                                       5, (0, 255, 255), -1)
                            cv2.putText(vis_image, "B1",
                                        (px(first_bottom_idx) - 15, py(prices[first_bottom_idx]) + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                            # Middle peak
                            cv2.circle(vis_image,
                                       (px(middle_peak_idx), py(prices[middle_peak_idx])),
                                       5, (0, 255, 255), -1)
                            cv2.putText(vis_image, "P",
                                        (px(middle_peak_idx) - 5, py(prices[middle_peak_idx]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                            # Second bottom
                            cv2.circle(vis_image,
                                       (px(last_bottom_idx), py(prices[last_bottom_idx])),
                                       5, (0, 255, 255), -1)
                            cv2.putText(vis_image, "B2",
                                        (px(last_bottom_idx) - 15, py(prices[last_bottom_idx]) + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                            # Draw neckline
                            draw_dashed_line(vis_image,
                                     (px(first_bottom_idx), py(prices[middle_peak_idx])),
                                     (px(last_bottom_idx), py(prices[middle_peak_idx])),
                                     (0, 255, 255), 2)

                            # Draw breakout point if it exists
                            if prices[-1] > prices[middle_peak_idx]:
                                # Find where price crosses above the middle peak
                                for i in range(middle_peak_idx, len(prices)):
                                    if prices[i] > prices[middle_peak_idx]:
                                        breakout_idx = i
                                        break
                                else:
                                    breakout_idx = len(prices) - 1

                                cv2.circle(vis_image,
                                           (px(breakout_idx), py(prices[breakout_idx])),
                                           5, (0, 255, 0), -1)

                                # Draw breakout arrow
                                cv2.arrowedLine(vis_image,
                                                (px(breakout_idx), py(prices[breakout_idx]) - 20),
                                                (px(breakout_idx), py(prices[breakout_idx]) - 5),
                                                (0, 255, 0), 2, tipLength=0.3)

            elif "Head and Shoulders" in pattern_name:
                # Similar logic to draw head and shoulders pattern
                # Implementation details similar to above patterns
                pass

    # Save the result
    cv2.imwrite(output_filename, vis_image)
    print(f"Analysis complete. Result saved as '{output_filename}'")


def get_pattern_description(pattern_name):
    """Return description and trading strategy for detected pattern"""
    descriptions = {
        "Ascending Triangle": {
            "description": "A bullish continuation pattern characterized by a flat upper resistance line and a rising lower support line, indicating accumulation before a breakout.",
            "signal": "BUY when price breaks above resistance line. Set stop loss below the most recent low.",
            "target": "Measure the height of the triangle at its widest point and project that distance from the breakout point."
        },
        "Double Bottom": {
            "description": "A bullish reversal pattern formed by two price lows at approximately the same level, separated by a moderate price peak.",
            "signal": "BUY when price breaks above the peak between the two bottoms. Set stop loss below the second bottom.",
            "target": "Measure the distance from the bottom to the middle peak and project that distance from the breakout point."
        },
        "Head and Shoulders": {
            "description": "A bearish reversal pattern consisting of three peaks, with the middle peak (head) higher than the two surrounding peaks (shoulders).",
            "signal": "SELL when price breaks below the neckline. Set stop loss above the right shoulder.",
            "target": "Measure the distance from the head to the neckline and project that distance from the breakdown point."
        }
    }

    # Extract base pattern name (remove trend information)
    base_pattern = pattern_name.split(':')[0].strip()

    return descriptions.get(base_pattern, {
        "description": "Pattern detected with insufficient specific information.",
        "signal": "Monitor for confirmation before trading.",
        "target": "Use standard risk management techniques."
    })
# ============= Main Execution =============

if __name__ == "__main__":
    # Example usage
    image_dir = "test_charts/new"
    json_dir = "test_charts/new"

    # Check if directories exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"Created directory: {image_dir}")

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
        print(f"Created directory: {json_dir}")

    # Get a sample image path
    sample_images = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.startswith('marked_')]

    if sample_images:
        sample_image = os.path.join(image_dir, sample_images[0])
        print(f"Analyzing sample image: {sample_image}")
        prices, patterns = analyze_chart(sample_image, image_dir=image_dir, json_dir=json_dir)
        print("Detected patterns:", patterns)
    else:
        print(f"No images found in {image_dir}. Please add some .png images.")

# Test pattern detection with synthetic data
if __name__ == "__main__":
    print("\nTesting pattern detection with synthetic data:")

    # Double Bottom test data
    double_bottom_data = np.array([50, 48, 45, 43, 45, 48, 46, 43, 42, 45, 48, 52, 55, 58])
    print("Double Bottom test:")
    patterns = detect_technical_patterns(double_bottom_data)
    print("Detected patterns:", patterns)

    # Head and Shoulders test data
    head_shoulders_data = np.array([50, 52, 54, 52, 50, 52, 56, 60, 56, 52, 54, 58, 54, 50])
    print("\nHead and Shoulders test:")
    patterns = detect_technical_patterns(head_shoulders_data)
    print("Detected patterns:", patterns)

    # Ascending Triangle test data
    ascending_triangle_data = np.array([50, 52, 48, 50, 49, 52, 50, 52, 51, 52, 52, 53, 54, 55])
    print("\nAscending Triangle test:")
    patterns = detect_technical_patterns(ascending_triangle_data)
    print("Detected patterns:", patterns)