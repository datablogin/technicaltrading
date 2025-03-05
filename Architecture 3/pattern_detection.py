import numpy as np
from scipy.signal import find_peaks


def detect_peaks_and_troughs(prices, prominence=0.05, distance=5):
    """Detect peaks and troughs in price data with adaptive parameters"""
    # Normalize prices for consistent detection
    norm_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))

    # Find peaks with prominence filtering
    peaks, peak_props = find_peaks(norm_prices, prominence=prominence, distance=distance)

    # Find troughs by inverting the signal
    troughs, trough_props = find_peaks(-norm_prices, prominence=prominence, distance=distance)

    return peaks, troughs, peak_props, trough_props


def detect_double_bottom(prices, peaks, troughs):
    """Detect double bottom pattern (W-shape)"""
    if len(troughs) < 2 or len(peaks) < 1:
        print("Not enough troughs or peaks for double bottom")
        return False

    # Find all potential double bottoms
    for i in range(len(troughs) - 1):
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

            # Troughs should be at similar levels and peak should be higher
            if trough_diff / (trough_avg + 0.0001) < 0.15 and peak_val > trough_avg * 1.05:
                print(f"Double bottom detected: Troughs at {t1},{t2}, peak at {middle_peak}")
                return True

    return False


def detect_head_and_shoulders(prices, peaks, troughs):
    """Detect head and shoulders pattern"""
    if len(peaks) < 3:
        print("Not enough peaks for head and shoulders")
        return False

    # Need at least 3 consecutive peaks
    for i in range(len(peaks) - 2):
        p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]

        # Check for minimum distance between peaks
        if p2 - p1 < 2 or p3 - p2 < 2:
            continue

        # Check if middle peak (head) is higher than shoulders
        if prices[p2] > prices[p1] and prices[p2] > prices[p3]:
            # Check if shoulders are at similar levels
            shoulder_diff = abs(prices[p1] - prices[p3])
            shoulder_avg = (prices[p1] + prices[p3]) / 2

            if shoulder_diff / (shoulder_avg + 0.0001) < 0.15:
                print(f"Head and shoulders detected: Shoulders at {p1},{p3}, head at {p2}")
                return True

    return False


def detect_triangle(prices, peaks, troughs):
    """Detect triangle patterns (ascending, descending, symmetric)"""
    if len(peaks) < 2 or len(troughs) < 2:
        print("Not enough peaks or troughs for triangle")
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

        print(f"Peak trend: {peak_trend:.4f}, Trough trend: {trough_trend:.4f}")

        # Determine triangle type based on trend directions
        if peak_trend < -0.01 and trough_trend > 0.01:
            print("Symmetric Triangle detected")
            return "Symmetric Triangle"
        elif peak_trend < -0.01 and abs(trough_trend) < 0.01:
            print("Descending Triangle detected")
            return "Descending Triangle"
        elif abs(peak_trend) < 0.01 and trough_trend > 0.01:
            print("Ascending Triangle detected")
            return "Ascending Triangle"

    return None


def detect_technical_patterns(prices):
    """Main function to detect all technical patterns"""
    # Multi-scale detection for different pattern sizes
    results = {}

    # Use different window sizes for multi-scale detection
    window_sizes = [3, 5, 8]

    for window in window_sizes:
        prominence = 0.05 if window == 3 else (0.1 if window == 5 else 0.15)

        # Get peaks and troughs
        peaks, troughs, _, _ = detect_peaks_and_troughs(prices, prominence=prominence, distance=window)

        print(f"Window size {window}: Found {len(peaks)} peaks and {len(troughs)} troughs")

        # Check for patterns
        if detect_double_bottom(prices, peaks, troughs):
            results['Double Bottom'] = True

        if detect_head_and_shoulders(prices, peaks, troughs):
            results['Head and Shoulders'] = True

        triangle_type = detect_triangle(prices, peaks, troughs)
        if triangle_type:
            results[triangle_type] = True

    return results