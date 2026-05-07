"""
Drusen Detector - Main Detection Algorithm
===========================================

Implements automated drusen detection in fundus images using:
- Green channel extraction
- CLAHE contrast enhancement
- Otsu's thresholding
- Morphological operations
- Contour analysis
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrusenDetector:
    """Automated drusen detection in fundus images."""
    
    def __init__(self, min_area=30, max_area=500, clahe_clip=2.0, blur_kernel=5):
        """
        Initialize detector with parameters.
        
        Args:
            min_area: Minimum drusen area in pixels (removes noise)
            max_area: Maximum drusen area in pixels (removes large artifacts)
            clahe_clip: CLAHE contrast enhancement strength
            blur_kernel: Gaussian blur kernel size
        """
        self.min_area = min_area
        self.max_area = max_area
        self.clahe_clip = clahe_clip
        self.blur_kernel = blur_kernel
    
    def detect(self, image_path):
        """
        Detect drusen in a single fundus image.
        
        Args:
            image_path: Path to fundus image file
            
        Returns:
            dict with:
                - marked_image: Image with marked drusen
                - drusen_count: Number of drusen detected
                - drusen_list: List of (x, y, radius) for each drusen
                - classification: "drusen" or "no_drusen"
                - preprocessing_steps: Intermediate images for visualization
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return {'error': f'Could not load image: {image_path}'}
            
            # Step 1: Extract green channel
            green = self._extract_green_channel(img)
            
            # Step 2: Preprocess
            preprocessed = self._preprocess(green)
            
            # Step 3: Segment
            binary = self._segment(preprocessed)
            
            # Step 4: Clean morphologically
            cleaned = self._morphological_cleanup(binary)
            
            # Step 5: Find drusen
            drusen = self._find_drusen(cleaned)
            
            # Step 6: Mark on original image
            marked = self._mark_drusen(img, drusen)
            
            # Classify
            classification = "drusen" if len(drusen) > 0 else "no_drusen"
            
            return {
                'marked_image': marked,
                'drusen_count': len(drusen),
                'drusen_list': drusen,
                'classification': classification,
                'green_channel': green,
                'preprocessed': preprocessed,
                'binary': binary,
                'cleaned': cleaned
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {'error': str(e)}
    
    def _extract_green_channel(self, img_bgr):
        """
        Extract green channel from BGR image.
        
        Green channel provides best contrast for drusen in fundus images.
        This is a medical imaging standard.
        
        Args:
            img_bgr: Image in BGR color space (OpenCV format)
            
        Returns:
            Green channel as grayscale image
        """
        # OpenCV uses BGR, so green is index 1
        green = img_bgr[:, :, 1]
        return green
    
    def _preprocess(self, gray_img):
        """
        Preprocess image: normalize, enhance contrast, smooth.
        
        Args:
            gray_img: Grayscale image
            
        Returns:
            Preprocessed grayscale image
        """
        # Step 1: Normalize intensity to 0-255 range
        normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # CLAHE is more sophisticated than simple histogram equalization
        # It works on local regions and prevents over-amplification
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # Step 3: Apply Gaussian blur to reduce noise
        # Gentle blur preserves edges while removing noise
        blurred = cv2.GaussianBlur(enhanced, (self.blur_kernel, self.blur_kernel), 1.0)
        
        return blurred
    
    def _segment(self, gray_img):
        """
        Segment bright regions using Otsu's thresholding.
        
        Otsu's method automatically finds the threshold that maximizes
        between-class variance. No manual threshold tuning needed.
        
        Args:
            gray_img: Preprocessed grayscale image
            
        Returns:
            Binary image (0 or 255)
        """
        # Use Otsu's method for automatic threshold computation
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _morphological_cleanup(self, binary_img):
        """
        Clean up binary image using morphological operations.
        
        Opens (erode then dilate) to remove small noise while preserving drusen.
        
        Args:
            binary_img: Binary segmentation image
            
        Returns:
            Cleaned binary image
        """
        # Create morphological structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Opening: erode then dilate
        # Removes small bright regions (noise) while preserving larger objects (drusen)
        opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    def _find_drusen(self, binary_img):
        """
        Find individual drusen by contour detection and filtering.
        
        Args:
            binary_img: Cleaned binary image
            
        Returns:
            List of (x, y, radius) tuples for detected drusen
        """
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        drusen = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by size
            if self.min_area <= area <= self.max_area:
                # Fit circle to contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                drusen.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'area': area
                })
        
        return drusen
    
    def _mark_drusen(self, original_img, drusen_list):
        """
        Mark detected drusen on original image.
        
        Args:
            original_img: Original BGR fundus image
            drusen_list: List of detected drusen
            
        Returns:
            Marked image with red circles around drusen
        """
        marked = original_img.copy()
        
        # Draw circles for each drusen
        for drusen in drusen_list:
            center = drusen['center']
            radius = drusen['radius']
            
            # Draw red circle (BGR format: (B, G, R))
            cv2.circle(marked, center, radius, (0, 0, 255), 2)
            
            # Draw center point
            cv2.circle(marked, center, 3, (0, 0, 255), -1)
        
        return marked
    
    def process_directory(self, image_dir, output_dir=None, label="unknown"):
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing fundus images
            output_dir: Directory to save marked images (optional)
            label: Label for this batch (e.g., "healthy", "drusen")
            
        Returns:
            Dictionary with results for all images
        """
        image_dir = Path(image_dir)
        
        results = {
            'label': label,
            'images': {}
        }
        
        # Supported image formats
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"No images found in {image_dir}")
            return results
        
        # Process each image
        for i, image_path in enumerate(sorted(image_files), 1):
            logger.info(f"[{label}] Processing {i}/{len(image_files)}: {image_path.name}")
            
            result = self.detect(str(image_path))
            
            if 'error' in result:
                results['images'][image_path.name] = result
                continue
            
            # Save marked image
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_name = f"marked_{label}_{image_path.stem}.jpg"
                output_path = output_dir / output_name
                
                cv2.imwrite(str(output_path), result['marked_image'])
                result['output_path'] = str(output_path)
            
            # Store result (without large image arrays for memory efficiency)
            results['images'][image_path.name] = {
                'drusen_count': result['drusen_count'],
                'classification': result['classification'],
                'drusen_list': result['drusen_list'],
                'output_path': result.get('output_path', None)
            }
        
        return results


if __name__ == "__main__":
    # Example usage
    detector = DrusenDetector()
    
    # Test single image
    result = detector.detect("data/raw/drusen/sample.jpg")
    if 'error' not in result:
        print(f"Drusen found: {result['drusen_count']}")
    
    # Process entire directory
    results = detector.process_directory("data/raw/drusen/", output_dir="results/marked_images/")
    print(f"Processed {len(results['images'])} images")
