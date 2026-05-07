"""
Main Processing Script - Full Drusen Detection Pipeline
========================================================

This script processes all fundus images in data/raw/ and generates:
1. Marked images with detected drusen
2. Evaluation metrics (Accuracy, Sensitivity, Specificity)
3. Detailed JSON and CSV reports

Usage:
    python scripts/process_dataset.py
"""

import sys
from pathlib import Path
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from drusen_detector import DrusenDetector
from evaluation import DrusenEvaluator


def main():
    """Main processing pipeline."""
    
    # Setup paths
    data_dir = Path("data/raw")
    results_dir = Path("results")
    marked_dir = results_dir / "marked_images"
    reports_dir = results_dir / "reports"
    
    # Create output directories
    marked_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("DRUSEN DETECTION - FULL PIPELINE")
    print("=" * 70)
    
    # Initialize detector and evaluator
    detector = DrusenDetector(
        min_area=30,
        max_area=500,
        clahe_clip=2.0,
        blur_kernel=5
    )
    
    evaluator = DrusenEvaluator()
    
    # Step 1: Load ground truth
    print("\n[1/4] Loading ground truth labels...")
    healthy_dir = data_dir / "healthy"
    drusen_dir = data_dir / "drusen"
    
    if not healthy_dir.exists() or not drusen_dir.exists():
        print(f"ERROR: Data directories not found!")
        print(f"  Expected: {healthy_dir}")
        print(f"  Expected: {drusen_dir}")
        return
    
    evaluator.load_ground_truth(str(healthy_dir), str(drusen_dir))
    
    # Step 2: Process healthy images
    print("\n[2/4] Processing healthy fundus images...")
    healthy_results = detector.process_directory(
        str(healthy_dir),
        output_dir=str(marked_dir),
        label="healthy"
    )
    
    healthy_count = len(healthy_results['images'])
    print(f"  ✓ Processed {healthy_count} healthy images")
    
    # Step 3: Process drusen images
    print("\n[3/4] Processing fundus images with drusen...")
    drusen_results = detector.process_directory(
        str(drusen_dir),
        output_dir=str(marked_dir),
        label="drusen"
    )
    
    drusen_count = len(drusen_results['images'])
    print(f"  ✓ Processed {drusen_count} drusen images")
    
    # Step 4: Add detections to evaluator
    print("\n[4/4] Evaluating performance...")
    
    # Add healthy detections
    for filename, result in healthy_results['images'].items():
        if 'error' not in result:
            evaluator.add_detection(
                filename,
                result['classification'],
                result['drusen_count']
            )
    
    # Add drusen detections
    for filename, result in drusen_results['images'].items():
        if 'error' not in result:
            evaluator.add_detection(
                filename,
                result['classification'],
                result['drusen_count']
            )
    
    # Generate reports
    print("\n" + evaluator.get_report())
    
    # Save JSON report
    json_report = reports_dir / "evaluation_report.json"
    evaluator.save_report(str(json_report))
    
    # Save detailed CSV
    csv_report = reports_dir / "detailed_results.csv"
    evaluator.save_detailed_results(str(csv_report))
    
    # Print misclassification summary
    misclass = evaluator.get_misclassifications()
    
    if misclass['false_positives']:
        print("\n⚠ FALSE POSITIVES (Healthy misclassified as Drusen):")
        for fp in misclass['false_positives'][:5]:
            print(f"  - {fp}")
        if len(misclass['false_positives']) > 5:
            print(f"  ... and {len(misclass['false_positives']) - 5} more")
    
    if misclass['false_negatives']:
        print("\n⚠ FALSE NEGATIVES (Drusen misclassified as Healthy):")
        for fn in misclass['false_negatives'][:5]:
            print(f"  - {fn}")
        if len(misclass['false_negatives']) > 5:
            print(f"  ... and {len(misclass['false_negatives']) - 5} more")
    
    # Summary statistics
    total_drusen_detected = sum(
        r['drusen_count'] for r in drusen_results['images'].values() 
        if 'drusen_count' in r
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total images processed: {healthy_count + drusen_count}")
    print(f"  - Healthy: {healthy_count}")
    print(f"  - Drusen: {drusen_count}")
    print(f"Total drusen detected: {total_drusen_detected}")
    print(f"Average drusen per drusen image: {total_drusen_detected / drusen_count:.1f}")
    print(f"\nResults saved to: {results_dir.absolute()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
