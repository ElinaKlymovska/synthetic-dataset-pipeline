#!/usr/bin/env python3
"""
Dataset validation script.
Validates the quality and consistency of generated character images.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import load_config, setup_logging, load_image, calculate_image_similarity
from identity_checker import IdentityChecker


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate generated character dataset")
    
    parser.add_argument(
        "--reference", 
        default="data/input/character_reference.jpg",
        help="Path to reference character image"
    )
    
    parser.add_argument(
        "--dataset-dir", 
        default="data/output",
        help="Directory containing generated images"
    )
    
    parser.add_argument(
        "--config", 
        default="config/generation_config.yaml",
        help="Path to generation configuration file"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="data/metadata",
        help="Output directory for validation reports"
    )
    
    parser.add_argument(
        "--identity-threshold", 
        type=float, 
        default=0.7,
        help="Identity similarity threshold"
    )
    
    parser.add_argument(
        "--min-resolution", 
        type=int, 
        nargs=2, 
        default=[512, 512],
        help="Minimum image resolution (width height)"
    )
    
    parser.add_argument(
        "--max-file-size", 
        type=int, 
        default=10,
        help="Maximum file size in MB"
    )
    
    return parser.parse_args()


def load_generated_images(dataset_dir):
    """Load all generated images from the dataset directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    image_paths = []
    
    for file_path in Path(dataset_dir).rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            try:
                image = load_image(str(file_path))
                images.append(image)
                image_paths.append(str(file_path))
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    return images, image_paths


def check_image_quality(image, min_resolution, max_file_size_mb):
    """Check if image meets quality requirements."""
    height, width = image.shape[:2]
    
    # Check resolution
    if width < min_resolution[0] or height < min_resolution[1]:
        return False, f"Resolution {width}x{height} below minimum {min_resolution[0]}x{min_resolution[1]}"
    
    # Check file size (approximate)
    estimated_size_mb = (width * height * 3) / (1024 * 1024)
    if estimated_size_mb > max_file_size_mb:
        return False, f"Estimated size {estimated_size_mb:.1f}MB exceeds limit {max_file_size_mb}MB"
    
    return True, "OK"


def analyze_dataset_diversity(images, image_paths):
    """Analyze diversity of the generated dataset."""
    if len(images) < 2:
        return {"error": "Not enough images for diversity analysis"}
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            similarity = calculate_image_similarity(images[i], images[j])
            similarities.append(similarity)
    
    # Calculate diversity metrics
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    std_similarity = np.std(similarities)
    
    # Count unique images (similarity < 0.8)
    unique_count = sum(1 for s in similarities if s < 0.8)
    uniqueness_ratio = unique_count / len(similarities) if similarities else 0
    
    return {
        "total_images": len(images),
        "total_comparisons": len(similarities),
        "average_similarity": float(avg_similarity),
        "min_similarity": float(min_similarity),
        "max_similarity": float(max_similarity),
        "similarity_std": float(std_similarity),
        "unique_images": unique_count,
        "uniqueness_ratio": float(uniqueness_ratio)
    }


def create_validation_report(results, args):
    """Create a comprehensive validation report."""
    report_file = f"{args.output_dir}/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Add metadata
    results["validation_metadata"] = {
        "validation_date": datetime.now().isoformat(),
        "reference_image": args.reference,
        "dataset_directory": args.dataset_dir,
        "identity_threshold": args.identity_threshold,
        "min_resolution": args.min_resolution,
        "max_file_size_mb": args.max_file_size
    }
    
    # Save report
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return report_file


def create_visualization_report(images, image_paths, identity_scores, output_dir):
    """Create visualization of validation results."""
    if not images:
        return None
    
    # Create figure with subplots
    n_images = min(len(images), 12)  # Show max 12 images
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(n_images):
        # Show image
        axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        
        # Add title with identity score
        title = f"Image {i+1}"
        if i < len(identity_scores):
            score = identity_scores[i]
            color = 'green' if score >= 0.7 else 'red'
            title += f"\nScore: {score:.3f}"
            axes[i].set_title(title, color=color, fontsize=10)
        else:
            axes[i].set_title(title, fontsize=10)
        
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = f"{output_dir}/validation_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return viz_file


def print_summary(results):
    """Print a human-readable summary of validation results."""
    print("\n📊 Dataset Validation Summary")
    print("=" * 50)
    
    # Basic stats
    print(f"Total images: {results['total_images']}")
    print(f"Valid images: {results['valid_images']}")
    print(f"Invalid images: {results['invalid_images']}")
    print(f"Validation rate: {results['validation_rate']:.1%}")
    
    # Identity check results
    if 'identity_check' in results:
        identity = results['identity_check']
        print(f"\nIdentity Consistency:")
        print(f"  Passed identity check: {identity['passed_count']}")
        print(f"  Failed identity check: {identity['failed_count']}")
        print(f"  Average identity score: {identity['average_score']:.3f}")
        print(f"  Identity pass rate: {identity['pass_rate']:.1%}")
    
    # Quality check results
    if 'quality_check' in results:
        quality = results['quality_check']
        print(f"\nQuality Check:")
        print(f"  Resolution issues: {quality['resolution_issues']}")
        print(f"  File size issues: {quality['file_size_issues']}")
        print(f"  Quality pass rate: {quality['pass_rate']:.1%}")
    
    # Diversity analysis
    if 'diversity_analysis' in results:
        diversity = results['diversity_analysis']
        print(f"\nDiversity Analysis:")
        print(f"  Average similarity: {diversity['average_similarity']:.3f}")
        print(f"  Uniqueness ratio: {diversity['uniqueness_ratio']:.1%}")
        print(f"  Unique images: {diversity['unique_images']}/{diversity['total_comparisons']}")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    if results['validation_rate'] >= 0.9:
        print("  ✅ Excellent dataset quality")
    elif results['validation_rate'] >= 0.7:
        print("  ⚠️ Good dataset quality with some issues")
    else:
        print("  ❌ Poor dataset quality - needs improvement")


def main():
    """Main validation function."""
    args = parse_arguments()
    
    print("🔍 Character Dataset Validator")
    print("=" * 50)
    
    # Setup logging
    log_file = f"{args.output_dir}/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file, "INFO")
    
    try:
        # Load reference image
        if not os.path.exists(args.reference):
            print(f"❌ Reference image not found: {args.reference}")
            sys.exit(1)
        
        reference_image = load_image(args.reference)
        logger.info(f"Loaded reference image: {args.reference}")
        
        # Load generated images
        if not os.path.exists(args.dataset_dir):
            print(f"❌ Dataset directory not found: {args.dataset_dir}")
            sys.exit(1)
        
        images, image_paths = load_generated_images(args.dataset_dir)
        logger.info(f"Loaded {len(images)} generated images from {args.dataset_dir}")
        
        if not images:
            print("❌ No images found in dataset directory")
            sys.exit(1)
        
        # Initialize results
        results = {
            "total_images": len(images),
            "valid_images": 0,
            "invalid_images": 0,
            "validation_rate": 0.0
        }
        
        # Quality check
        logger.info("Performing quality checks...")
        quality_issues = []
        valid_images = []
        valid_paths = []
        
        for i, (image, path) in enumerate(zip(images, image_paths)):
            is_valid, message = check_image_quality(image, args.min_resolution, args.max_file_size)
            
            if is_valid:
                valid_images.append(image)
                valid_paths.append(path)
                results["valid_images"] += 1
            else:
                quality_issues.append({
                    "image": path,
                    "issue": message
                })
                results["invalid_images"] += 1
        
        results["validation_rate"] = results["valid_images"] / results["total_images"]
        results["quality_check"] = {
            "resolution_issues": sum(1 for issue in quality_issues if "Resolution" in issue["issue"]),
            "file_size_issues": sum(1 for issue in quality_issues if "size" in issue["issue"]),
            "pass_rate": results["validation_rate"],
            "issues": quality_issues
        }
        
        # Identity check
        if valid_images:
            logger.info("Performing identity consistency checks...")
            identity_checker = IdentityChecker(threshold=args.identity_threshold)
            
            identity_results = identity_checker.check_identity_consistency(
                reference_image, valid_images, valid_paths
            )
            
            if identity_results["success"]:
                passed_count = len(identity_results["passed_images"])
                failed_count = len(identity_results["failed_images"])
                
                results["identity_check"] = {
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                    "average_score": identity_results["average_score"],
                    "pass_rate": passed_count / len(valid_images) if valid_images else 0,
                    "scores": identity_results["scores"]
                }
                
                # Get identity scores for visualization
                identity_scores = identity_results["scores"]
            else:
                logger.warning("Identity check failed")
                identity_scores = [0.0] * len(valid_images)
        else:
            identity_scores = []
        
        # Diversity analysis
        if len(valid_images) >= 2:
            logger.info("Analyzing dataset diversity...")
            diversity_results = analyze_dataset_diversity(valid_images, valid_paths)
            results["diversity_analysis"] = diversity_results
        
        # Create reports
        report_file = create_validation_report(results, args)
        
        if valid_images:
            viz_file = create_visualization_report(valid_images, valid_paths, identity_scores, args.output_dir)
            logger.info(f"Visualization saved to {viz_file}")
        
        # Print summary
        print_summary(results)
        
        print(f"\n✅ Validation completed!")
        print(f"📄 Detailed report: {report_file}")
        if valid_images:
            print(f"🖼️ Visualization: {viz_file}")
        
        # Return exit code based on validation rate
        if results["validation_rate"] < 0.5:
            sys.exit(1)  # Exit with error if less than 50% valid
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 