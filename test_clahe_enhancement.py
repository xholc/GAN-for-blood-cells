#!/usr/bin/env python3
"""
Test script for CLAHE + Bit Plane WBC image enhancement.
Tests the enhancement on 5 WBC types: Neutrophil, Basophil, Monocyte, Lymphocyte, Eosinophil
"""

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utililies import (
    clahe_bit_plane_enhance,
    visualize_wbc_enhancement,
    test_wbc_samples
)


def test_single_image(image_path, cell_type):
    """Test enhancement on a single image and display results."""
    print(f"\n{'='*60}")
    print(f"Testing CLAHE + Bit Plane Enhancement: {cell_type}")
    print(f"{'='*60}")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    # Get all enhancement results
    results = clahe_bit_plane_enhance(image_path)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'{cell_type} - CLAHE + Bit Plane Enhancement', fontsize=16, fontweight='bold')

    # Row 1: Original and basic enhancements
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(results['original'])
    ax1.set_title('Original', fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(results['clahe_color'])
    ax2.set_title('CLAHE Color\n(LAB space)', fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(results['bit_plane_clahe'])
    ax3.set_title('Bit Plane + CLAHE\n(R4,G4,B5)', fontsize=12)
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(results['hsv_enhanced'])
    ax4.set_title('HSV Enhanced\n(Saturation boost)', fontsize=12)
    ax4.axis('off')

    # Row 2: Granule and nucleus focus
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(results['granule_enhanced'])
    ax5.set_title('Granule Detection\n(Laplacian + CLAHE)', fontsize=12)
    ax5.axis('off')

    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(results['blended_granule'])
    ax6.set_title('Blended Enhancement\n(70% CLAHE + 30% Granule)', fontsize=12)
    ax6.axis('off')

    ax7 = fig.add_subplot(2, 4, 7)
    ax7.imshow(results['nucleus_mask'], cmap='gray')
    ax7.set_title('Nucleus Mask\n(Otsu threshold)', fontsize=12)
    ax7.axis('off')

    # Comparison: Original vs Best Enhancement
    ax8 = fig.add_subplot(2, 4, 8)
    # Side-by-side comparison
    h, w = results['original'].shape[:2]
    comparison = np.zeros((h, w*2 + 10, 3), dtype=np.uint8)
    comparison[:, :w] = results['original']
    comparison[:, w+10:] = results['blended_granule']
    ax8.imshow(comparison)
    ax8.set_title('Comparison\n(Original | Enhanced)', fontsize=12)
    ax8.axis('off')

    plt.tight_layout()
    return fig


def test_all_wbc_samples(sample_dir):
    """Test all WBC samples in a directory."""
    cell_types = ['neutrophil', 'basophil', 'monocyte', 'lymphocyte', 'eosinophil']

    print("\n" + "="*60)
    print("CLAHE + Bit Plane Enhancement Test for WBC Classification")
    print("="*60)
    print("\nThis enhancement is designed to:")
    print("  - Enhance nucleus contrast (blue-purple staining)")
    print("  - Highlight cytoplasm features")
    print("  - Detect granules (critical for WBC discrimination)")
    print("\nGranule characteristics by cell type:")
    print("  - Neutrophil: Fine pink-purple granules")
    print("  - Eosinophil: Large orange-red granules")
    print("  - Basophil: Large dark purple granules")
    print("  - Monocyte: Fine azurophilic granules")
    print("  - Lymphocyte: Minimal to no granules")
    print("="*60)

    figures = []
    for cell_type in cell_types:
        # Try different file extensions
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            image_path = os.path.join(sample_dir, f"{cell_type}{ext}")
            if os.path.exists(image_path):
                fig = test_single_image(image_path, cell_type.capitalize())
                if fig:
                    figures.append(fig)
                break
        else:
            print(f"Warning: No image found for {cell_type}")

    return figures


def create_comparison_figure(sample_dir):
    """Create a single figure comparing all cell types."""
    cell_types = ['neutrophil', 'basophil', 'monocyte', 'lymphocyte', 'eosinophil']

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle('WBC CLAHE + Bit Plane Enhancement Comparison', fontsize=16, fontweight='bold')

    column_titles = ['Original', 'CLAHE Color', 'Granule Detection', 'Blended Enhancement']

    for row_idx, cell_type in enumerate(cell_types):
        # Find image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            path = os.path.join(sample_dir, f"{cell_type}{ext}")
            if os.path.exists(path):
                image_path = path
                break

        if image_path is None:
            for col_idx in range(4):
                axes[row_idx, col_idx].text(0.5, 0.5, 'Image not found',
                                           ha='center', va='center')
                axes[row_idx, col_idx].axis('off')
            continue

        # Get enhancement results
        results = clahe_bit_plane_enhance(image_path)

        # Plot results
        keys = ['original', 'clahe_color', 'granule_enhanced', 'blended_granule']
        for col_idx, key in enumerate(keys):
            axes[row_idx, col_idx].imshow(results[key])
            axes[row_idx, col_idx].axis('off')
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(column_titles[col_idx], fontsize=12)

        # Add cell type label
        axes[row_idx, 0].set_ylabel(cell_type.capitalize(), fontsize=12, rotation=0,
                                     labelpad=60, va='center')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Default sample directory
    sample_dir = os.path.join(os.path.dirname(__file__), "test_samples")

    # Check if sample directory exists
    if not os.path.exists(sample_dir):
        print(f"Creating sample directory: {sample_dir}")
        os.makedirs(sample_dir)
        print("\nPlease add your WBC sample images to:")
        print(f"  {sample_dir}")
        print("\nExpected filenames:")
        print("  - neutrophil.jpg (or .png)")
        print("  - basophil.jpg (or .png)")
        print("  - monocyte.jpg (or .png)")
        print("  - lymphocyte.jpg (or .png)")
        print("  - eosinophil.jpg (or .png)")
    else:
        # Test all samples
        figures = test_all_wbc_samples(sample_dir)

        # Create comparison figure if images exist
        if figures:
            comparison_fig = create_comparison_figure(sample_dir)
            print("\nDisplaying results...")
            plt.show()
        else:
            print("\nNo sample images found. Please add images to:")
            print(f"  {sample_dir}")
