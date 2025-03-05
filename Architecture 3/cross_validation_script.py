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
    classes = sorted(list(set(true_patterns + [p for p in detected_patterns if p])))

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
                                   output_dict=True)

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

    return df, report


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