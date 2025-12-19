#!/usr/bin/env python3
"""
Demo script for CLAHE + Bit Plane WBC image enhancement.
Creates synthetic WBC-like images to demonstrate the enhancement pipeline.
"""

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utililies import (
    clahe_bit_plane_enhance,
    apply_clahe,
    apply_clahe_color,
    get_bit_plane
)


def create_synthetic_wbc(cell_type, size=360):
    """
    Create a synthetic WBC image for demonstration.
    Simulates the key features of each cell type.
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 245  # Light pink background

    center = size // 2
    cell_radius = size // 3

    if cell_type == 'neutrophil':
        # Multi-lobed nucleus, fine pink-purple granules
        cv2.circle(img, (center, center), cell_radius, (220, 200, 230), -1)  # Light purple cytoplasm

        # Multi-lobed nucleus (3-5 lobes)
        nucleus_color = (120, 80, 140)  # Dark purple
        for offset in [(-20, -15), (10, -10), (25, 15)]:
            cv2.ellipse(img, (center + offset[0], center + offset[1]),
                       (25, 20), 0, 0, 360, nucleus_color, -1)

        # Fine granules
        for _ in range(80):
            x = center + np.random.randint(-cell_radius+20, cell_radius-20)
            y = center + np.random.randint(-cell_radius+20, cell_radius-20)
            cv2.circle(img, (x, y), 2, (180, 160, 200), -1)

    elif cell_type == 'eosinophil':
        # Bilobed nucleus, large orange-red granules
        cv2.circle(img, (center, center), cell_radius, (200, 180, 220), -1)

        # Bilobed nucleus
        nucleus_color = (100, 60, 130)
        cv2.ellipse(img, (center - 20, center), (25, 30), 0, 0, 360, nucleus_color, -1)
        cv2.ellipse(img, (center + 20, center), (25, 30), 0, 0, 360, nucleus_color, -1)

        # Large orange-red granules
        for _ in range(40):
            x = center + np.random.randint(-cell_radius+15, cell_radius-15)
            y = center + np.random.randint(-cell_radius+15, cell_radius-15)
            cv2.circle(img, (x, y), 5, (100, 120, 220), -1)  # Orange-red in BGR

    elif cell_type == 'basophil':
        # Large dark purple granules obscuring nucleus
        cv2.circle(img, (center, center), cell_radius, (190, 170, 200), -1)

        # Nucleus (often obscured)
        cv2.ellipse(img, (center, center), (40, 35), 0, 0, 360, (110, 70, 130), -1)

        # Large dark purple granules
        for _ in range(50):
            x = center + np.random.randint(-cell_radius+10, cell_radius-10)
            y = center + np.random.randint(-cell_radius+10, cell_radius-10)
            cv2.circle(img, (x, y), 6, (80, 40, 100), -1)

    elif cell_type == 'lymphocyte':
        # Large round nucleus, minimal cytoplasm, no granules
        cv2.circle(img, (center, center), cell_radius - 20, (210, 200, 230), -1)

        # Large nucleus taking most of cell
        cv2.circle(img, (center - 5, center), cell_radius - 35, (90, 60, 120), -1)

    elif cell_type == 'monocyte':
        # Kidney-shaped nucleus, grayish-blue cytoplasm, fine granules
        cv2.circle(img, (center, center), cell_radius + 10, (210, 195, 215), -1)

        # Kidney/horseshoe shaped nucleus
        cv2.ellipse(img, (center, center), (50, 40), 30, 0, 360, (100, 70, 130), -1)
        cv2.ellipse(img, (center + 10, center - 5), (25, 20), 0, 0, 360, (210, 195, 215), -1)

        # Fine azurophilic granules
        for _ in range(30):
            x = center + np.random.randint(-cell_radius, cell_radius)
            y = center + np.random.randint(-cell_radius, cell_radius)
            cv2.circle(img, (x, y), 2, (160, 140, 180), -1)

    # Add some RBCs in background (pale pink circles)
    for pos in [(50, 50), (size-50, 50), (50, size-50), (size-50, size-50),
                (50, center), (size-50, center)]:
        cv2.circle(img, pos, 35, (200, 200, 230), -1)
        cv2.circle(img, pos, 15, (220, 210, 235), -1)

    return img


def demonstrate_enhancement():
    """Demonstrate the CLAHE + Bit Plane enhancement on synthetic images."""

    print("\n" + "="*70)
    print("CLAHE + Bit Plane Enhancement Demo for WBC Classification")
    print("="*70)
    print("\nKey features targeted:")
    print("  - Nucleus: Enhanced contrast using CLAHE on LAB/blue channel")
    print("  - Cytoplasm: Better visibility through bit plane extraction")
    print("  - Granules: Detected using Laplacian + CLAHE combination")
    print("\nCell type characteristics:")
    print("  - Neutrophil: Multi-lobed nucleus, fine pink-purple granules")
    print("  - Eosinophil: Bilobed nucleus, large orange-red granules")
    print("  - Basophil: Obscured nucleus, large dark purple granules")
    print("  - Lymphocyte: Large round nucleus, minimal cytoplasm")
    print("  - Monocyte: Kidney-shaped nucleus, fine azurophilic granules")
    print("="*70)

    cell_types = ['neutrophil', 'eosinophil', 'basophil', 'lymphocyte', 'monocyte']

    # Create output directory
    sample_dir = os.path.join(os.path.dirname(__file__), "test_samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Create and save synthetic images
    print("\nGenerating synthetic WBC images...")
    for cell_type in cell_types:
        img = create_synthetic_wbc(cell_type)
        save_path = os.path.join(sample_dir, f"{cell_type}.png")
        cv2.imwrite(save_path, img)
        print(f"  Saved: {save_path}")

    # Create comparison figure
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle('CLAHE + Bit Plane Enhancement for WBC Classification\n'
                 'Enhancing Nucleus, Cytoplasm, and Granule Features',
                 fontsize=16, fontweight='bold')

    column_titles = ['Original', 'CLAHE Color', 'Bit Plane+CLAHE', 'Granule Detection', 'Blended']

    for row_idx, cell_type in enumerate(cell_types):
        image_path = os.path.join(sample_dir, f"{cell_type}.png")

        # Get enhancement results
        results = clahe_bit_plane_enhance(image_path)

        # Plot results
        keys = ['original', 'clahe_color', 'bit_plane_clahe', 'granule_enhanced', 'blended_granule']
        for col_idx, key in enumerate(keys):
            axes[row_idx, col_idx].imshow(results[key])
            axes[row_idx, col_idx].axis('off')
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(column_titles[col_idx], fontsize=12, fontweight='bold')

        # Add cell type label
        axes[row_idx, 0].text(-0.15, 0.5, cell_type.capitalize(),
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=12, fontweight='bold',
                             va='center', ha='right')

    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    plt.savefig(os.path.join(sample_dir, 'enhancement_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison figure to: {os.path.join(sample_dir, 'enhancement_comparison.png')}")

    # Show detailed view for each cell type
    print("\nGenerating detailed views...")
    for cell_type in cell_types:
        image_path = os.path.join(sample_dir, f"{cell_type}.png")
        results = clahe_bit_plane_enhance(image_path)

        fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
        fig2.suptitle(f'{cell_type.capitalize()} - Detailed Enhancement View', fontsize=14, fontweight='bold')

        # Row 1
        axes2[0, 0].imshow(results['original'])
        axes2[0, 0].set_title('Original')
        axes2[0, 0].axis('off')

        axes2[0, 1].imshow(results['clahe_color'])
        axes2[0, 1].set_title('CLAHE (LAB)')
        axes2[0, 1].axis('off')

        axes2[0, 2].imshow(results['bit_plane_clahe'])
        axes2[0, 2].set_title('Bit Plane + CLAHE')
        axes2[0, 2].axis('off')

        axes2[0, 3].imshow(results['hsv_enhanced'])
        axes2[0, 3].set_title('HSV Enhanced')
        axes2[0, 3].axis('off')

        # Row 2
        axes2[1, 0].imshow(results['granule_enhanced'])
        axes2[1, 0].set_title('Granule Detection')
        axes2[1, 0].axis('off')

        axes2[1, 1].imshow(results['blended_granule'])
        axes2[1, 1].set_title('Blended Enhancement')
        axes2[1, 1].axis('off')

        axes2[1, 2].imshow(results['nucleus_mask'], cmap='gray')
        axes2[1, 2].set_title('Nucleus Mask')
        axes2[1, 2].axis('off')

        # Side by side comparison
        h, w = results['original'].shape[:2]
        comparison = np.zeros((h, w*2 + 10, 3), dtype=np.uint8)
        comparison[:, :w] = results['original']
        comparison[:, w+10:] = results['blended_granule']
        axes2[1, 3].imshow(comparison)
        axes2[1, 3].set_title('Original | Enhanced')
        axes2[1, 3].axis('off')

        plt.tight_layout()

    print("\nDisplaying results...")
    plt.show()


if __name__ == "__main__":
    demonstrate_enhancement()
