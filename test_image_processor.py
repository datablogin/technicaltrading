# test_image_processor.py
import cv2
import numpy as np
from image_processor import extract_price_data, detect_pattern, markup_image


def test_image_processor(image_path):
    print(f"\nTesting image: {image_path}")

    # Test extract_price_data
    try:
        prices = extract_price_data(image_path)
        print(f"Extracted prices (first 10 and last 10): {prices[:10]}...{prices[-10:]}")

        # Check for price variation
        if np.max(prices) == np.min(prices):
            print("Warning: No variation in prices - using default range")
        else:
            print(f"Min price: {np.min(prices)}, Max price: {np.max(prices)}")

        # Test detect_pattern
        troughs, peaks = detect_pattern(prices)
        print(f"Troughs found: {troughs}")
        print(f"Peaks found: {peaks}")

        # Optional: Visualize the marked-up image (save to disk)
        marked_image = markup_image(image_path, prices, troughs, peaks)  # Ensure markup_image is imported or defined
        output_path = f"marked_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, marked_image)
        print(f"Marked image saved to: {output_path}")

    except Exception as e:
        print(f"Error processing image: {str(e)}")
    # Add this to test_image_processor.py after each test
    with open("test_results.txt", "a") as f:
        f.write(f"\nTesting image: {image_path}\n")
        f.write(f"Extracted prices: {prices[:10]}...{prices[-10:]}\n")
        f.write(f"Min price: {np.min(prices)}, Max price: {np.max(prices)}\n")
        f.write(f"Troughs found: {troughs}\n")
        f.write(f"Peaks found: {peaks}\n")
        f.write(f"Marked image saved to: {output_path}\n")


if __name__ == "__main__":
    import os

    # List of image paths to test (update these paths to your actual image locations)
    test_images = [
        "test_charts/buy_double_bottom.png",
        "test_charts/sell_double_top.png",
        "test_charts/hold_flat.png"
    ]

    # Ensure the images exist
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_path}")
        else:
            test_image_processor(image_path)