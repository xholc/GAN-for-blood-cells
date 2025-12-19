import cv2
import numpy as np
from matplotlib import pyplot as plt

# TensorFlow is optional - only needed for generate_images function
try:
    import tensorflow as tf
except ImportError:
    tf = None

##### CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to a single channel image.

    Args:
        image: Single channel grayscale image (uint8)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_clahe_color(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to a color image in LAB color space.
    This enhances contrast while preserving color information.

    Args:
        image: BGR color image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE enhanced color image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # Merge channels back
    lab_clahe = cv2.merge([l_clahe, a, b])

    # Convert back to BGR
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def clahe_bit_plane_enhance(image_path, r_bit=4, g_bit=4, b_bit=5,
                            clip_limit=3.0, tile_grid_size=(8, 8),
                            enhance_granules=True):
    """
    Combine bit plane extraction with CLAHE for enhanced WBC visualization.

    This method is designed to enhance:
    - Nucleus contrast (typically dark purple/blue)
    - Cytoplasm visibility
    - Granule details (important for WBC type discrimination)

    Blood cell staining (Giemsa/Wright's stain):
    - Nucleus: Blue-purple (high in blue channel)
    - Eosinophil granules: Orange-red (high in red channel)
    - Basophil granules: Dark purple (high in blue channel)
    - Neutrophil granules: Fine pink-purple
    - Monocyte: Grayish-blue cytoplasm with fine granules
    - Lymphocyte: Blue cytoplasm, minimal granules

    Args:
        image_path: Path to the input image
        r_bit, g_bit, b_bit: Bit planes for each channel (0-7)
        clip_limit: CLAHE clip limit (higher = more contrast)
        tile_grid_size: CLAHE tile grid size
        enhance_granules: Whether to apply additional granule enhancement

    Returns:
        Dictionary containing various enhanced images
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = {}

    # Original image
    results['original'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Split into BGR channels
    b, g, r = cv2.split(img)

    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)

    # CLAHE enhanced full color image
    clahe_color = apply_clahe_color(img, clip_limit, tile_grid_size)
    results['clahe_color'] = cv2.cvtColor(clahe_color, cv2.COLOR_BGR2RGB)

    # Extract bit planes from CLAHE-enhanced channels
    bit_plane_r = get_bit_plane(r_clahe, r_bit)
    bit_plane_g = get_bit_plane(g_clahe, g_bit)
    bit_plane_b = get_bit_plane(b_clahe, b_bit)

    # Apply scaling similar to original function
    kr = (bit_plane_r + 1) * (2**r_bit) - 1
    kg = (bit_plane_g + 1) * (2**g_bit) - 1
    kb = (bit_plane_b + 1) * (2**b_bit) - 1

    # Combine into RGB image
    bit_plane_result = np.stack((kr, kg, kb), axis=-1).astype(np.uint8)
    results['bit_plane_clahe'] = bit_plane_result

    # Granule enhancement using high-pass filtering on specific channels
    if enhance_granules:
        # For granule detection, we focus on local variations
        # Use Laplacian to detect edges and fine structures
        laplacian_r = cv2.Laplacian(r_clahe, cv2.CV_64F)
        laplacian_g = cv2.Laplacian(g_clahe, cv2.CV_64F)
        laplacian_b = cv2.Laplacian(b_clahe, cv2.CV_64F)

        # Normalize and convert
        laplacian_r = np.uint8(np.abs(laplacian_r))
        laplacian_g = np.uint8(np.abs(laplacian_g))
        laplacian_b = np.uint8(np.abs(laplacian_b))

        # Apply CLAHE to Laplacian output
        lap_r_clahe = clahe.apply(laplacian_r)
        lap_g_clahe = clahe.apply(laplacian_g)
        lap_b_clahe = clahe.apply(laplacian_b)

        # Granule enhanced image (emphasizes texture)
        granule_enhanced = np.stack((lap_r_clahe, lap_g_clahe, lap_b_clahe), axis=-1)
        results['granule_enhanced'] = granule_enhanced

        # Blend original CLAHE with granule enhancement
        alpha = 0.7
        blended = cv2.addWeighted(
            cv2.cvtColor(clahe_color, cv2.COLOR_BGR2RGB), alpha,
            granule_enhanced, 1-alpha, 0
        )
        results['blended_granule'] = blended

    # Nucleus enhancement (focus on blue channel)
    # Nucleus typically stains darker in blue channel
    nucleus_mask = b_clahe.copy()
    _, nucleus_binary = cv2.threshold(nucleus_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results['nucleus_mask'] = nucleus_binary

    # Create a visualization that highlights both nucleus and granules
    hsv = cv2.cvtColor(clahe_color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Enhance saturation for better granule visibility
    s_enhanced = clahe.apply(s)
    v_enhanced = clahe.apply(v)

    hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
    results['hsv_enhanced'] = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    return results


def visualize_wbc_enhancement(image_path, cell_type="Unknown", save_path=None):
    """
    Visualize WBC enhancement results for a single image.

    Args:
        image_path: Path to WBC image
        cell_type: Type of WBC for display
        save_path: Optional path to save the figure
    """
    # Get enhanced results
    results = clahe_bit_plane_enhance(image_path)

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'WBC Enhancement: {cell_type}', fontsize=14, fontweight='bold')

    # Plot results
    titles = [
        ('original', 'Original'),
        ('clahe_color', 'CLAHE Color'),
        ('bit_plane_clahe', 'Bit Plane + CLAHE'),
        ('hsv_enhanced', 'HSV Enhanced'),
        ('granule_enhanced', 'Granule Detection'),
        ('blended_granule', 'Blended Enhancement'),
        ('nucleus_mask', 'Nucleus Mask'),
    ]

    for idx, (key, title) in enumerate(titles):
        row = idx // 4
        col = idx % 4
        if key in results:
            if key == 'nucleus_mask':
                axes[row, col].imshow(results[key], cmap='gray')
            else:
                axes[row, col].imshow(results[key])
            axes[row, col].set_title(title)
            axes[row, col].axis('off')

    # Hide empty subplot
    axes[1, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def test_wbc_samples(image_paths, cell_types):
    """
    Test CLAHE bit plane enhancement on multiple WBC samples.

    Args:
        image_paths: List of image paths
        cell_types: List of corresponding cell types
    """
    n_samples = len(image_paths)

    for img_path, cell_type in zip(image_paths, cell_types):
        print(f"\nProcessing {cell_type}...")
        visualize_wbc_enhancement(img_path, cell_type)

    plt.show()


##### bit planes
def get_bit_plane(image, bit):
    # Extract the bit plane
    bit_plane = np.bitwise_and(image, 2**bit)
    # Scale it to 0 or 255
    bit_plane = np.array(bit_plane * 255 / 2**bit).astype(np.uint8)

    return bit_plane

def get_transform_image(image, rb, gb, bb):
    img = cv2.imread(image)
    # Split the image into BGR channels
    b, g, r = cv2.split(img)
    bit_plane_r = get_bit_plane(r, rb) #4
    kr = (bit_plane_r+1)* (2**rb)-1
    bit_plane_g = get_bit_plane(g, gb) #4
    kg = (bit_plane_g+1)* (2**gb)-1
    bit_plane_b = get_bit_plane(b, bb) #5
    kb = (bit_plane_b+1)* (2**bb)-1
    result_fig = np.stack((kr, kg, kb), axis=-1)

    return result_fig


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

#test function
'''
for example_input, example_target in test_dataset.take(1):
    generate_images(generator, example_input, example_target)  '''
    
    