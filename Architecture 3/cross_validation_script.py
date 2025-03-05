import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import your pattern detection functions
from chart_analyzer import analyze_chart


def run_cross_validation(dataset_path="chart_validation/dataset.json"):
    """Run cross-validation on the generated dataset"""
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    results = []

    # Process each chart
    for i, item in enumerate(dataset):
        print(f"Processing chart {i + 1}/{len(dataset)}: {item['pattern']}")

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
        results.append({
            "image": os.path.basename(image_path),
            "true_pattern": item['pattern'],
            "detected_pattern": primary_pattern,
            "confidence": max_confidence,
            "correct": item['pattern'] == primary_pattern,
            "all_detected": detected_patterns
        })

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Calculate accuracy
    accuracy = df['correct'].mean()
    print(f"Overall accuracy: {accuracy:.2f}")

    # Generate confusion matrix
    true_patterns = df['true_pattern'].tolist()
    detected_patterns = df['detected_pattern'].tolist()

    # Get unique pattern classes
    classes = sorted(list(set([p for p in true_patterns if p] + [p for p in detected_patterns if p])))

    # Create confusion matrix
    cm = confusion_matrix(true_patterns,
                          [d if d in classes else "None" for d in detected_patterns],
                          labels=classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Pattern')
    plt.ylabel('True Pattern')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("chart_validation/confusion_matrix.png")

    # Generate classification report
    report = classification_report(true_patterns,
                                   [d if d in classes else "None" for d in detected_patterns],
                                   labels=classes,
                                   output_dict=True,
                                   zero_division=0)  # Add zero_division parameter to prevent warnings

    # Save results
    df.to_csv("chart_validation/validation_results.csv", index=False)

    # Save report
    with open("chart_validation/classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # Plot confidence distribution by pattern
    plt.figure(figsize=(12, 6))
    for pattern in classes:
        pattern_df = df[df['true_pattern'] == pattern]
        if not pattern_df.empty:
            sns.kdeplot(pattern_df['confidence'], label=pattern)

    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution by Pattern')
    plt.legend()
    plt.savefig("chart_validation/confidence_distribution.png")

    # Create detailed visualization of results
    visualize_validation_results(dataset_path, df)

    return df, report


# Add this function to your cross_validation_script.py
def visualize_validation_results(dataset_path, results_df):
    """Create visual summary of pattern detection accuracy with fixed layout"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import os

    # Load dataset info to find image paths
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Create a lookup dictionary of image paths
    image_lookup = {os.path.basename(item['image']): item['image'] for item in dataset}

    # Create figure with proper layout - use GridSpec for more control
    plt.figure(figsize=(16, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, figure=plt.gcf())

    # Get unique classes
    true_patterns = results_df['true_pattern'].tolist()
    detected_patterns = [d if d else "None" for d in results_df['detected_pattern'].tolist()]
    classes = sorted(list(set([p for p in true_patterns if p] + [p for p in detected_patterns if p and p != "None"])))
    if "None" not in classes:
        classes.append("None")

    # Create confusion matrix
    cm = confusion_matrix(
        true_patterns,
        detected_patterns,
        labels=classes
    )

    # Plot confusion matrix in larger space
    ax_cm = plt.subplot(gs[0, :])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes, ax=ax_cm)
    ax_cm.set_xlabel('Predicted Pattern')
    ax_cm.set_ylabel('True Pattern')
    ax_cm.set_title('Confusion Matrix')

    # Pattern types to display
    pattern_types = ["Ascending Triangle", "Double Bottom", "Head and Shoulders"]

    # Plot examples in a 2x3 grid below the confusion matrix
    for i, pattern in enumerate(pattern_types):
        # Find correct detection example (top row)
        correct = results_df[(results_df['true_pattern'] == pattern) &
                             (results_df['detected_pattern'] == pattern)]

        if not correct.empty:
            example = correct.iloc[0]
            img_path = image_lookup.get(example['image'])
            if img_path and os.path.exists(img_path):
                ax = plt.subplot(gs[1, i])
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"Correct {pattern}\nConf: {example['confidence']:.2f}")
                ax.axis('off')
        else:
            # Create empty subplot to maintain grid
            ax = plt.subplot(gs[1, i])
            ax.text(0.5, 0.5, "No correct examples", ha='center', va='center')
            ax.set_title(f"Correct {pattern}")
            ax.axis('off')

        # Find incorrect detection example (bottom row)
        incorrect = results_df[(results_df['true_pattern'] == pattern) &
                               (results_df['detected_pattern'] != pattern)]

        if not incorrect.empty:
            example = incorrect.iloc[0]
            img_path = image_lookup.get(example['image'])
            if img_path and os.path.exists(img_path):
                ax = plt.subplot(gs[2, i])
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"Missed {pattern}\nDetected: {example['detected_pattern']}")
                ax.axis('off')
        else:
            # Create empty subplot to maintain grid
            ax = plt.subplot(gs[2, i])
            ax.text(0.5, 0.5, "No incorrect examples", ha='center', va='center')
            ax.set_title(f"Missed {pattern}")
            ax.axis('off')

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig("chart_validation/validation_summary.png", bbox_inches='tight')
    print(f"Visualization saved to chart_validation/validation_summary.png")
    plt.close()




if __name__ == "__main__":
    df, report = run_cross_validation()

    # Print summary
    print("\nClassification Report:")
    for cls, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"\n{cls}:")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1-score: {metrics['f1-score']:.2f}")
            print(f"  Support: {metrics['support']}")

