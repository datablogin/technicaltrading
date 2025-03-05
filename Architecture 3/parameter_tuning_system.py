import numpy as np
from scipy.optimize import minimize
from chart_analyzer import analyze_chart



def optimize_detection_parameters(validation_dataset, initial_params=None):
    """Optimize pattern detection parameters to maximize accuracy"""
    if initial_params is None:
        # Default initial parameters
        initial_params = {
            "double_bottom_threshold": 0.6,
            "head_shoulders_threshold": 0.6,
            "triangle_threshold": 0.6,
            "prominence_factor_small": 0.02,
            "prominence_factor_medium": 0.04,
            "prominence_factor_large": 0.06
        }

    # Convert parameters to a simple array for optimization
    param_names = list(initial_params.keys())
    param_values = np.array([initial_params[name] for name in param_names])

    # Define objective function (negative accuracy)
    def objective(params):
        # Convert params back to dict
        param_dict = {name: params[i] for i, name in enumerate(param_names)}

        # Update detection thresholds
        update_detection_parameters(param_dict)

        # Run validation
        results = []

        # Process each chart
        for item in validation_dataset:
            # Analyze chart
            image_path = item['image']
            _, detected_patterns = analyze_chart(image_path)

            # Extract primary pattern (non-trend)
            primary_pattern = None
            max_confidence = 0

            for pattern, confidence in detected_patterns.items():
                if "Trend" not in pattern and float(confidence) > max_confidence:
                    primary_pattern = pattern
                    max_confidence = float(confidence)

            # Record result
            correct = item['pattern'] == primary_pattern
            results.append(correct)

        # Calculate accuracy
        accuracy = np.mean(results)

        # Return negative accuracy (for minimization)
        return -accuracy

    # Define parameter bounds
    bounds = [
        (0.2, 0.9),  # double_bottom_threshold
        (0.2, 0.9),  # head_shoulders_threshold
        (0.2, 0.9),  # triangle_threshold
        (0.01, 0.1),  # prominence_factor_small
        (0.01, 0.1),  # prominence_factor_medium
        (0.01, 0.1)  # prominence_factor_large
    ]

    # Run optimization
    result = minimize(objective, param_values, bounds=bounds, method='L-BFGS-B')

    # Get optimized parameters
    optimized_params = {name: result.x[i] for i, name in enumerate(param_names)}

    print("Optimization complete!")
    print(f"Initial parameters: {initial_params}")
    print(f"Optimized parameters: {optimized_params}")
    print(f"Accuracy improvement: {-result.fun - objective(param_values):.4f}")

    return optimized_params


def update_detection_parameters(params):
    """Update the pattern detection functions with new parameters"""
    # This function updates the global variables or function parameters
    # used by your pattern detection system
    global double_bottom_threshold, head_shoulders_threshold, triangle_threshold
    global prominence_factor_small, prominence_factor_medium, prominence_factor_large

    # Update thresholds
    double_bottom_threshold = params["double_bottom_threshold"]
    head_shoulders_threshold = params["head_shoulders_threshold"]
    triangle_threshold = params["triangle_threshold"]

    # Update prominence factors
    prominence_factor_small = params["prominence_factor_small"]
    prominence_factor_medium = params["prominence_factor_medium"]
    prominence_factor_large = params["prominence_factor_large"]