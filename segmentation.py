import datetime
from skimage.color import label2rgb
from skimage.filters import sobel
from scipy import ndimage as ndi
import os
from skimage import io
import matplotlib.pyplot as plt  # Use pyplot for plotting
import numpy as np
import skimage as ski
from skimage.filters import gaussian, threshold_otsu, rank
from skimage.morphology import disk, binary_erosion, binary_dilation, local_minima, local_maxima
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import minimum_filter, maximum_filter

# image path
image_path = 'images/'

for image_file in os.listdir(image_path):

    if not (image_file.lower().endswith('.jpg') or image_file.lower().endswith('.png')):
        continue  # skip non-image files

    print("Processing image file: " + image_file)
    image_full_path = os.path.join(image_path, image_file)

    # Get base filename without extension
    base_name = os.path.splitext(image_file)[0]
    file_extension = os.path.splitext(image_file)[1]
    
    # Create subdivisions directory if it doesn't exist
    subdivisions_dir = os.path.join(image_path, 'subdivisions')
    os.makedirs(subdivisions_dir, exist_ok=True)
    
    # Check if subdivisions already exist for this image
    expected_subdivisions = []
    for i in range(1, 5):
        subdivision_filename = f"{base_name}_subdivision_{i}{file_extension}"
        expected_subdivisions.append(subdivision_filename)
    
    # Check if all expected subdivisions exist
    subdivisions_exist = all(
        os.path.exists(os.path.join(subdivisions_dir, sub_file)) 
        for sub_file in expected_subdivisions
    )
    
    if subdivisions_exist:
        print(f"  Subdivisions for {image_file} already exist. Skipping subdivision creation.")
        continue  # Skip to next image
    
    # subdivide the image and convert to grayscale
    image = io.imread(image_full_path, as_gray=True)  # Load image as grayscale
    
    # Get image dimensions
    height, width = image.shape
    
    # Calculate dimensions for 2x2 grid (4 subdivisions)
    sub_height = height // 2
    sub_width = width // 2
    
    # Process each of the 4 subdivisions
    subdivision_count = 1
    for row in range(2):
        for col in range(2):
            # Calculate crop boundaries
            start_row = row * sub_height
            end_row = start_row + sub_height if row == 0 else height  # Handle remainder for bottom row
            start_col = col * sub_width
            end_col = start_col + sub_width if col == 0 else width    # Handle remainder for right column
            
            # Extract sub-image
            sub_image = image[start_row:end_row, start_col:end_col]
            
            # Create filename for subdivision
            subdivision_filename = f"{base_name}_subdivision_{subdivision_count}{file_extension}"
            subdivision_path = os.path.join(subdivisions_dir, subdivision_filename)
            
            # Save the subdivision
            plt.imsave(subdivision_path, sub_image, cmap='gray')
            print(f"  Saved subdivision {subdivision_count}: {subdivision_filename}")
            
            subdivision_count += 1

# Now process all subdivision images with segmentation
subdivisions_dir = os.path.join(image_path, 'subdivisions')
if os.path.exists(subdivisions_dir):
    print("\nProcessing subdivision images...")
    
    # Get only the subdivision files (not the processed ones)
    subdivision_files = [f for f in os.listdir(subdivisions_dir) 
                        if (f.lower().endswith('.jpg') or f.lower().endswith('.png')) 
                        and '_subdivision_' in f 
                        and not f.startswith('overlay_') 
                        and not f.startswith('elevation_') 
                        and not f.startswith('segmentation_')]
    
    for subdivision_file in subdivision_files:
        print(f"Processing subdivision: {subdivision_file}")
        subdivision_full_path = os.path.join(subdivisions_dir, subdivision_file)
        
        # Check if processing results already exist
        overlay_filename = os.path.join(subdivisions_dir, f'overlay_{subdivision_file}')
        if os.path.exists(overlay_filename):
            print(f"  Processing results for {subdivision_file} already exist. Skipping.")
            continue
        
        # Load the subdivision image
        sub_image = io.imread(subdivision_full_path, as_gray=True)

        # Enhanced edge detection and segmentation
        # Light smoothing to reduce noise but preserve edges
        sub_image_smooth = gaussian(sub_image, sigma=0.3)  # Reduced from 0.5

        # Enhanced elevation map using multiple edge detectors
        from skimage.filters import scharr, prewitt
        elevation_sobel = sobel(sub_image_smooth)
        elevation_scharr = scharr(sub_image_smooth)
        
        # Combine edge detectors for more sensitive edge detection
        elevation_map = np.maximum(elevation_sobel, elevation_scharr)
        
        # Enhance the elevation map contrast
        elevation_map = (elevation_map - elevation_map.min()) / (elevation_map.max() - elevation_map.min())
        elevation_map = np.power(elevation_map, 0.8)  # Gamma correction to enhance edges

        # Create edge-based markers using local maxima and minima
        footprint = np.ones((3, 3), dtype=bool)  # 3x3 square
        
        # Find local minima in original image (dark regions)
        local_min_mask = (sub_image_smooth == minimum_filter(sub_image_smooth, footprint=footprint))
        
        # Find local maxima in original image (bright regions) 
        local_max_mask = (sub_image_smooth == maximum_filter(sub_image_smooth, footprint=footprint))
        
        # Create more sensitive threshold-based markers
        thresh = threshold_otsu(sub_image_smooth)
        
        # Use more aggressive thresholds to create more segments
        very_dark = sub_image_smooth < (thresh * 0.5)  # Much darker regions
        very_bright = sub_image_smooth > (thresh * 1.5)  # Much brighter regions
        
        # Combine different marker strategies
        markers = np.zeros_like(sub_image_smooth, dtype=int)
        
        # Background markers (very dark areas or local minima)
        background_regions = very_dark | local_min_mask
        markers[background_regions] = 1
        
        # Foreground markers (very bright areas or local maxima)
        foreground_regions = very_bright | local_max_mask
        markers[foreground_regions] = 2
        
        # Add edge-based markers for better boundary detection
        # Find strong edges in elevation map
        edge_threshold = np.percentile(elevation_map, 85)  # Top 15% of edges
        strong_edges = elevation_map > edge_threshold
        
        # Clean up markers with minimal morphological operations to preserve detail
        small_element = disk(1)
        
        # Light cleanup of background markers
        background_mask = (markers == 1)
        background_clean = binary_erosion(background_mask, small_element)
        
        # Light cleanup of foreground markers
        foreground_mask = (markers == 2)
        foreground_clean = binary_erosion(foreground_mask, small_element)
        
        # Rebuild markers
        markers = np.zeros_like(sub_image_smooth, dtype=int)
        markers[background_clean] = 1
        markers[foreground_clean] = 2
        
        # Ensure we have enough markers for detailed segmentation
        if np.sum(markers == 1) < 10:  # If too few background markers
            # Add more conservative background markers
            conservative_bg = sub_image_smooth < (thresh * 0.8)
            markers[conservative_bg] = 1
            
        if np.sum(markers == 2) < 10:  # If too few foreground markers
            # Add more conservative foreground markers
            conservative_fg = sub_image_smooth > (thresh * 1.2)
            markers[conservative_fg] = 2

        # Apply watershed segmentation with enhanced elevation map
        segmentation_rocks = watershed(elevation_map, markers, compactness=0.1)  # Lower compactness for more detailed boundaries
        
        # Minimal post-processing to preserve detail
        segmentation_rocks = ndi.binary_fill_holes(segmentation_rocks - 1)
        labeled_rocks, num_labels = ndi.label(segmentation_rocks)
        
        # Remove very small objects (noise) but keep detailed boundaries
        min_object_size = 20  # Reduced from default to keep smaller details
        for label_id in range(1, num_labels + 1):
            if np.sum(labeled_rocks == label_id) < min_object_size:
                labeled_rocks[labeled_rocks == label_id] = 0
        
        # Re-label after cleaning
        labeled_rocks, num_labels = ndi.label(labeled_rocks > 0)
        
        # Calculate object sizes and categorize them
        object_sizes = []
        for label_id in range(1, num_labels + 1):
            size = np.sum(labeled_rocks == label_id)
            object_sizes.append(size)
        
        if len(object_sizes) > 0:
            # Calculate size thresholds for 3 categories
            sizes_array = np.array(object_sizes)
            small_threshold = np.percentile(sizes_array, 33.33)  # Bottom 1/3
            large_threshold = np.percentile(sizes_array, 66.67)  # Top 1/3
            
            # Create size-based label image
            size_labeled_image = np.zeros_like(labeled_rocks)
            
            for label_id in range(1, num_labels + 1):
                size = np.sum(labeled_rocks == label_id)
                object_mask = (labeled_rocks == label_id)
                
                if size <= small_threshold:
                    size_labeled_image[object_mask] = 1  # Small objects
                elif size <= large_threshold:
                    size_labeled_image[object_mask] = 2  # Medium objects
                else:
                    size_labeled_image[object_mask] = 3  # Large objects
            
            # Define colors for each size category
            colors = [
                [0, 0, 0],        # Background (black)
                [1, 0, 0],        # Small objects (red)
                [0, 1, 0],        # Medium objects (green)
                [0, 0, 1]         # Large objects (blue)
            ]
            
            # Create the overlay with size-based coloring
            image_label_overlay = ski.color.label2rgb(
                size_labeled_image, 
                image=sub_image, 
                colors=colors,
                bg_label=0,
                alpha=0.6  # Slightly more transparent to see underlying detail
            )
            
            # Print size statistics
            small_count = np.sum(sizes_array <= small_threshold)
            medium_count = np.sum((sizes_array > small_threshold) & (sizes_array <= large_threshold))
            large_count = np.sum(sizes_array > large_threshold)
            
            print(f"  Object counts - Small (red): {small_count}, Medium (green): {medium_count}, Large (blue): {large_count}")
            print(f"  Size thresholds - Small: ≤{small_threshold:.0f}, Medium: {small_threshold:.0f}-{large_threshold:.0f}, Large: >{large_threshold:.0f}")
        
        else:
            # Fallback if no objects detected
            image_label_overlay = ski.color.label2rgb(labeled_rocks, image=sub_image, bg_label=0)
            print("  No objects detected, using default coloring")

        # Create visualization for subdivision
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Original image
        axes[0,0].imshow(sub_image, cmap=plt.cm.gray)
        axes[0,0].set_title(f'Original - {subdivision_file}')
        axes[0,0].set_axis_off()
        
        # Enhanced elevation map
        axes[0,1].imshow(elevation_map, cmap=plt.cm.gray)
        axes[0,1].set_title('Enhanced Elevation Map')
        axes[0,1].set_axis_off()

        # Segmentation boundaries
        axes[1,0].imshow(sub_image, cmap=plt.cm.gray)
        axes[1,0].contour(labeled_rocks, levels=np.arange(0.5, num_labels+0.5), linewidths=0.8, colors='yellow')
        axes[1,0].set_title(f'Detailed Segmentation ({num_labels} objects)')
        axes[1,0].set_axis_off()

        # Size-based overlay
        axes[1,1].imshow(image_label_overlay)
        axes[1,1].set_title('Size-Based Overlay\n(Red=Small, Green=Medium, Blue=Large)')
        axes[1,1].set_axis_off()

        fig.tight_layout()
        plt.show()

        # Save the segmentation result for subdivision
        plt.imsave(overlay_filename, image_label_overlay)
        print(f"  Saved overlay: overlay_{subdivision_file}")
        
        # Save enhanced elevation map
        elevation_filename = os.path.join(subdivisions_dir, f'elevation_{subdivision_file}')
        plt.imsave(elevation_filename, elevation_map, cmap='gray')
        print(f"  Saved elevation map: elevation_{subdivision_file}")

print("\nAll processing complete!")