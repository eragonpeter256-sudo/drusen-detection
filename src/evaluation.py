"""
Drusen Evaluator - Performance Metrics and Analysis
====================================================

Computes and reports:
- Accuracy, Sensitivity, Specificity, Precision, F1-Score
- Confusion matrix
- False positive/negative analysis
- JSON and CSV report generation
"""

import json
import csv
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrusenEvaluator:
    """Evaluate drusen detection performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.ground_truth = {}  # filename -> "healthy" or "drusen"
        self.detections = {}    # filename -> {"classification": ..., "drusen_count": ...}
    
    def load_ground_truth(self, healthy_dir, drusen_dir):
        """
        Load ground truth labels from directory structure.
        
        Args:
            healthy_dir: Directory containing healthy images
            drusen_dir: Directory containing drusen images
        """
        healthy_dir = Path(healthy_dir)
        drusen_dir = Path(drusen_dir)
        
        # Load healthy images
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        for ext in extensions:
            for img_path in healthy_dir.glob(f'*{ext}'):
                self.ground_truth[img_path.name] = 'healthy'
            for img_path in healthy_dir.glob(f'*{ext.upper()}'):
                self.ground_truth[img_path.name] = 'healthy'
        
        # Load drusen images
        for ext in extensions:
            for img_path in drusen_dir.glob(f'*{ext}'):
                self.ground_truth[img_path.name] = 'drusen'
            for img_path in drusen_dir.glob(f'*{ext.upper()}'):
                self.ground_truth[img_path.name] = 'drusen'
        
        logger.info(f"Loaded ground truth for {len(self.ground_truth)} images")
        logger.info(f"  Healthy: {sum(1 for v in self.ground_truth.values() if v == 'healthy')}")
        logger.info(f"  Drusen: {sum(1 for v in self.ground_truth.values() if v == 'drusen')}")
    
    def add_detection(self, filename, classification, drusen_count):
        """
        Add detection result for an image.
        
        Args:
            filename: Image filename
            classification: "drusen" or "no_drusen"
            drusen_count: Number of drusen detected
        """
        # Convert "no_drusen" to "healthy" for consistency
        if classification == "no_drusen":
            classification = "healthy"
        
        self.detections[filename] = {
            'classification': classification,
            'drusen_count': drusen_count
        }
    
    def _compute_metrics(self):
        """
        Compute classification metrics.
        
        Returns:
            dict with TP, TN, FP, FN, accuracy, sensitivity, specificity, etc.
        """
        tp = tn = fp = fn = 0
        
        for filename, detection in self.detections.items():
            if filename not in self.ground_truth:
                logger.warning(f"No ground truth for {filename}")
                continue
            
            truth = self.ground_truth[filename]
            pred = detection['classification']
            
            # True Positive: Correctly detected drusen
            if truth == 'drusen' and pred == 'drusen':
                tp += 1
            # True Negative: Correctly identified as healthy
            elif truth == 'healthy' and pred == 'healthy':
                tn += 1
            # False Positive: Healthy misclassified as drusen
            elif truth == 'healthy' and pred == 'drusen':
                fp += 1
            # False Negative: Drusen misclassified as healthy
            elif truth == 'drusen' and pred == 'healthy':
                fn += 1
        
        # Compute metrics
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        
        # Sensitivity: True Positive Rate (recall for positive class)
        # How many drusen cases are correctly identified
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity: True Negative Rate
        # How many healthy cases are correctly identified
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Precision: How many detected drusen are actually drusen
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # F1-Score: Harmonic mean of precision and recall
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) \
            if (precision + sensitivity) > 0 else 0
        
        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total': total,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1
        }
    
    def get_misclassifications(self):
        """
        Get lists of misclassified images.
        
        Returns:
            dict with 'false_positives' and 'false_negatives' lists
        """
        false_positives = []  # Healthy misclassified as drusen
        false_negatives = []  # Drusen misclassified as healthy
        
        for filename, detection in self.detections.items():
            if filename not in self.ground_truth:
                continue
            
            truth = self.ground_truth[filename]
            pred = detection['classification']
            
            if truth == 'healthy' and pred == 'drusen':
                false_positives.append(filename)
            elif truth == 'drusen' and pred == 'healthy':
                false_negatives.append(filename)
        
        return {
            'false_positives': sorted(false_positives),
            'false_negatives': sorted(false_negatives)
        }
    
    def get_report(self):
        """
        Generate a formatted performance report.
        
        Returns:
            Formatted string report
        """
        metrics = self._compute_metrics()
        
        report = "\n" + "=" * 70 + "\n"
        report += "DRUSEN DETECTION EVALUATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        report += "OVERALL PERFORMANCE\n"
        report += "-" * 70 + "\n"
        report += f"  Accuracy:       {metrics['accuracy']:.2f}  ({metrics['accuracy']*100:.2f}%)\n"
        report += f"  Sensitivity:    {metrics['sensitivity']:.2f}  ({metrics['sensitivity']*100:.2f}%)  [Recall for drusen]\n"
        report += f"  Specificity:    {metrics['specificity']:.2f}  ({metrics['specificity']*100:.2f}%)  [Recall for healthy]\n"
        report += f"  Precision:      {metrics['precision']:.2f}  ({metrics['precision']*100:.2f}%)\n"
        report += f"  F1-Score:       {metrics['f1']:.2f}\n\n"
        
        report += "CONFUSION MATRIX\n"
        report += "-" * 70 + "\n"
        report += "                    Predicted Healthy    Predicted Drusen\n"
        report += f"  Actual Healthy        {metrics['tn']:>6}              {metrics['fp']:>6}\n"
        report += f"  Actual Drusen         {metrics['fn']:>6}              {metrics['tp']:>6}\n\n"
        
        return report
    
    def print_report(self):
        """Print the performance report to console."""
        report = self.get_report()
        print(report)
        return report
    
    def save_report(self, output_path):
        """
        Save evaluation report as JSON.
        
        Args:
            output_path: Path to save JSON report
        """
        metrics = self._compute_metrics()
        misclass = self.get_misclassifications()
        
        report_data = {
            'metrics': {
                'accuracy': metrics['accuracy'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'precision': metrics['precision'],
                'f1_score': metrics['f1'],
                'total_images': metrics['total']
            },
            'confusion_matrix': {
                'true_positives': metrics['tp'],
                'true_negatives': metrics['tn'],
                'false_positives': metrics['fp'],
                'false_negatives': metrics['fn']
            },
            'misclassifications': {
                'false_positives': misclass['false_positives'],
                'false_negatives': misclass['false_negatives']
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
    
    def save_detailed_results(self, output_path):
        """
        Save detailed per-image results as CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Filename',
                'Ground Truth',
                'Predicted',
                'Drusen Count',
                'Correct',
                'Error Type'
            ])
            
            for filename, detection in sorted(self.detections.items()):
                if filename not in self.ground_truth:
                    continue
                
                truth = self.ground_truth[filename]
                pred = detection['classification']
                drusen_count = detection['drusen_count']
                
                correct = truth == pred
                
                if correct:
                    error_type = "OK"
                elif truth == 'healthy' and pred == 'drusen':
                    error_type = "False Positive"
                elif truth == 'drusen' and pred == 'healthy':
                    error_type = "False Negative"
                else:
                    error_type = "Unknown"
                
                writer.writerow([
                    filename,
                    truth,
                    pred,
                    drusen_count,
                    correct,
                    error_type
                ])
        
        logger.info(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = DrusenEvaluator()
    
    # Load ground truth
    evaluator.load_ground_truth("data/raw/healthy/", "data/raw/drusen/")
    
    # Add some example detections
    evaluator.add_detection("healthy_001.jpg", "healthy", 0)
    evaluator.add_detection("drusen_001.jpg", "drusen", 3)
    
    # Print report
    evaluator.print_report()
