"""
Simple Demo - Tests the colorization system with a sample design
"""

import cv2
import numpy as np
from design_colorizer import DesignColorizationSystem


def create_sample_design(width=800, height=800):
    """Create a simple black & white design for testing"""
    # White background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Black color for lines
    black = (0, 0, 0)
    thickness = 3
    
    # Draw a mandala-like design
    center = (width // 2, height // 2)
    
    # Outer circle
    cv2.circle(img, center, 350, black, thickness)
    
    # Inner circles
    for radius in [300, 250, 200, 150, 100, 50]:
        cv2.circle(img, center, radius, black, thickness)
    
    # Radial lines
    for i in range(12):
        angle = np.radians(i * 30)
        x_end = int(center[0] + 350 * np.cos(angle))
        y_end = int(center[1] + 350 * np.sin(angle))
        cv2.line(img, center, (x_end, y_end), black, thickness)
    
    # Corner decorations
    corner_radius = 60
    corners = [
        (80, 80), (width-80, 80),
        (80, height-80), (width-80, height-80)
    ]
    
    for corner in corners:
        cv2.circle(img, corner, corner_radius, black, thickness)
        cv2.circle(img, corner, corner_radius//2, black, thickness)
        cv2.line(img, (corner[0]-corner_radius, corner[1]), 
                (corner[0]+corner_radius, corner[1]), black, thickness)
        cv2.line(img, (corner[0], corner[1]-corner_radius), 
                (corner[0], corner[1]+corner_radius), black, thickness)
    
    # Border
    cv2.rectangle(img, (20, 20), (width-20, height-20), black, thickness)
    
    return img


if __name__ == "__main__":
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "AI Design Colorization - Demo" + " "*24 + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Create sample design
    print("Creating sample black & white design...")
    design = create_sample_design(800, 800)
    design_path = '737-SCAN-NEW.tif'
    cv2.imwrite(design_path, design)
    print(f"✓ Saved sample design: {design_path}\n")
    
    # Initialize the colorization system
    print("Initializing colorization system...")
    system = DesignColorizationSystem(
        min_region_area=100,      # Minimum size of regions to color
        gradient_probability=0.4  # 40% chance of gradients
    )
    print("✓ System ready!\n")
    
    # Process the design
    print("Starting colorization process...\n")
    output_paths = system.process(
        design_path=design_path,
        output_dir='./colorized_outputs',
        num_variations=6
    )
    
    print("\n✓ Complete! Check the './colorized_outputs' folder for results.")
    print(f"  - {len(output_paths)} colorized variations")
    print(f"  - Segmentation preview")
    print(f"  - Color palette swatches")
    print("\n" + "="*70 + "\n")