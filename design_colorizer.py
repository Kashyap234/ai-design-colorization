"""
AI Design Colorization System - Enhanced Version
Takes black & white designs and adds beautiful colors with GRADIENTS
Author: Enhanced for user requirements
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import colorsys
from typing import List, Tuple, Dict
import os
import gc  # For garbage collection


class ColorTheoryEngine:
    """Generates aesthetically pleasing color palettes using color theory"""
    
    def __init__(self):
        # Color harmony schemes with their angle relationships
        self.harmony_schemes = {
            'complementary': [0, 180],
            'analogous': [0, 30, 60],
            'triadic': [0, 120, 240],
            'split_complementary': [0, 150, 210],
            'tetradic': [0, 90, 180, 270],
            'square': [0, 90, 180, 270],
        }
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB (0-255 range)"""
        rgb_float = colorsys.hsv_to_rgb(h, s, v)
        return tuple(int(x * 255) for x in rgb_float)
    
    def rgb_to_hsv(self, r, g, b):
        """Convert RGB to HSV (0-1 range)"""
        return colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    
    def generate_palette(self, base_hue=None, scheme='triadic', num_colors=8):
        """
        Generate a harmonious color palette
        base_hue: 0-1, if None will be random
        scheme: harmony scheme name
        num_colors: total colors in palette
        """
        if base_hue is None:
            base_hue = np.random.random()
        
        # Get angles for this scheme
        angles = self.harmony_schemes.get(scheme, self.harmony_schemes['triadic'])
        
        palette = []
        
        # Generate base harmony colors
        for angle in angles:
            hue = (base_hue + angle/360.0) % 1.0
            
            # Create variations with different saturation and value
            variations_per_hue = max(1, num_colors // len(angles))
            
            for i in range(variations_per_hue):
                # Vary saturation and value for each variation
                if i == 0:
                    s, v = 0.8, 0.9  # Vibrant
                elif i == 1:
                    s, v = 0.6, 0.85  # Medium
                else:
                    s, v = 0.4 + np.random.random() * 0.4, 0.7 + np.random.random() * 0.2
                
                rgb = self.hsv_to_rgb(hue, s, v)
                palette.append(rgb)
        
        # Ensure we have exactly num_colors
        while len(palette) < num_colors:
            # Add neutral colors
            gray_val = int(np.random.random() * 156 + 100)  # 100-255
            palette.append((gray_val, gray_val, gray_val))
        
        return palette[:num_colors]
    
    def generate_multiple_palettes(self, num_palettes=6, colors_per_palette=8):
        """Generate multiple distinct color palettes"""
        palettes = []
        schemes = list(self.harmony_schemes.keys())
        
        for i in range(num_palettes):
            # Use different base hues and schemes
            base_hue = i / num_palettes
            scheme = schemes[i % len(schemes)]
            
            palette = self.generate_palette(base_hue, scheme, colors_per_palette)
            palettes.append(palette)
        
        return palettes


class DesignSegmenter:
    """Segments black & white design into distinct regions"""
    
    def __init__(self, min_area=50, max_dimension=3000):
        self.min_area = min_area
        self.max_dimension = max_dimension  # Maximum width or height
    
    def load_and_prepare(self, image_path):
        """Load image and convert to grayscale"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        original_shape = image.shape
        
        # Resize if image is too large (prevents memory errors)
        height, width = image.shape[:2]
        if width > self.max_dimension or height > self.max_dimension:
            # Calculate new dimensions maintaining aspect ratio
            scale = self.max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            print(f"      Resizing from {width}x{height} to {new_width}x{new_height} to prevent memory issues")
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        return image, gray
    
    def segment_design(self, gray_image):
        """
        Segment the design into regions
        Assumes black lines on white background
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Threshold to get binary image
        # Invert so regions to color are white
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, 
                                                cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and hierarchy
        # Keep only regions (holes in the design), not the outer boundaries
        valid_contours = []
        valid_hierarchy = []
        
        if hierarchy is not None:
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # Keep regions with sufficient area
                # Skip the outermost contour if it's too large (background)
                if area > self.min_area and area < binary.size * 0.5:
                    valid_contours.append(contour)
                    valid_hierarchy.append(hierarchy[0][idx])
        
        print(f"Found {len(valid_contours)} colorable regions")
        
        return valid_contours, binary
    
    def create_region_masks(self, image_shape, contours):
        """Create individual masks for each region (memory efficient)"""
        # Don't store all masks - we'll create them on-demand
        # Just return the shape info needed
        return {'shape': image_shape[:2], 'contours': contours}


class GradientGenerator:
    """Generates gradient fills for regions"""
    
    @staticmethod
    def create_linear_gradient(shape, color1, color2, angle=0):
        """
        Create a linear gradient
        angle: 0=horizontal, 90=vertical, etc.
        """
        height, width = shape[:2]
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Calculate gradient direction
        if angle == 0:  # Horizontal
            t = x / width
        elif angle == 90:  # Vertical
            t = y / height
        else:
            # General angle
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            # Project coordinates onto gradient direction
            x_centered = x - width / 2
            y_centered = y - height / 2
            
            proj = x_centered * cos_a + y_centered * sin_a
            proj_min = proj.min()
            proj_max = proj.max()
            
            t = (proj - proj_min) / (proj_max - proj_min)
        
        # Interpolate between colors
        for c in range(3):
            gradient[:, :, c] = (color1[c] * (1 - t) + color2[c] * t).astype(np.uint8)
        
        return gradient
    
    @staticmethod
    def create_radial_gradient(shape, color1, color2, center=None):
        """Create a radial gradient from center"""
        height, width = shape[:2]
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        
        if center is None:
            center = (width // 2, height // 2)
        
        # Create distance map from center
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        max_dist = distances.max()
        
        t = distances / max_dist
        
        # Interpolate between colors
        for c in range(3):
            gradient[:, :, c] = (color1[c] * (1 - t) + color2[c] * t).astype(np.uint8)
        
        return gradient


class SmartColorFiller:
    """Applies colors and gradients to design regions"""
    
    def __init__(self, gradient_probability=0.3):
        self.gradient_gen = GradientGenerator()
        self.gradient_probability = gradient_probability
    
    def calculate_region_properties(self, contour):
        """Calculate properties of a region"""
        area = cv2.contourArea(contour)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        return {
            'area': area,
            'centroid': (cx, cy),
            'bbox': (x, y, w, h),
            'aspect_ratio': float(w) / h if h > 0 else 1.0
        }
    
    def assign_colors_to_regions(self, contours, palette):
        """
        Assign colors to regions with good contrast
        Some regions get gradients, some get solid colors
        """
        region_fills = {}
        
        # Calculate properties
        region_props = [self.calculate_region_properties(cnt) for cnt in contours]
        
        # Sort by area (largest first)
        sorted_indices = sorted(range(len(region_props)), 
                               key=lambda i: region_props[i]['area'], 
                               reverse=True)
        
        used_colors = []
        
        for idx in sorted_indices:
            props = region_props[idx]
            
            # Decide if this region gets a gradient or solid color
            use_gradient = np.random.random() < self.gradient_probability
            
            # Select color(s)
            if not used_colors:
                color = palette[0]
            else:
                # Find color with maximum contrast to used colors
                available_colors = [c for c in palette if c not in used_colors]
                if not available_colors:
                    available_colors = palette
                
                # Calculate distances
                best_color = available_colors[0]
                max_dist = 0
                
                for color in available_colors:
                    min_dist = min([distance.euclidean(color, used) for used in used_colors])
                    if min_dist > max_dist:
                        max_dist = min_dist
                        best_color = color
                
                color = best_color
            
            if use_gradient and len(palette) > 1:
                # Pick a second color for gradient
                color2_candidates = [c for c in palette if c != color]
                if color2_candidates:
                    color2 = color2_candidates[np.random.randint(len(color2_candidates))]
                else:
                    color2 = palette[(palette.index(color) + 1) % len(palette)]
                
                # Choose gradient type
                gradient_type = np.random.choice(['linear', 'radial'])
                
                if gradient_type == 'linear':
                    angle = np.random.choice([0, 45, 90, 135])
                    region_fills[idx] = {
                        'type': 'linear_gradient',
                        'color1': color,
                        'color2': color2,
                        'angle': angle
                    }
                else:
                    region_fills[idx] = {
                        'type': 'radial_gradient',
                        'color1': color,
                        'color2': color2,
                        'center': props['centroid']
                    }
                
                used_colors.extend([color, color2])
            else:
                # Solid color
                region_fills[idx] = {
                    'type': 'solid',
                    'color': color
                }
                used_colors.append(color)
        
        return region_fills
    
    def apply_fills_to_image(self, image, contours, masks, region_fills):
        """Apply colors and gradients to the image (memory efficient)"""
        # Start with white background
        result = np.ones_like(image) * 255
        
        # Get shape info
        image_shape = masks['shape']
        stored_contours = masks['contours']
        
        # Apply each region's fill (create masks on-demand)
        for region_id, fill_info in region_fills.items():
            if region_id >= len(stored_contours):
                continue
            
            # Create mask on-demand (saves memory)
            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.drawContours(mask, [stored_contours[region_id]], -1, 255, -1)
            
            if fill_info['type'] == 'solid':
                # Solid color fill
                color = fill_info['color']
                result[mask == 255] = color[::-1]  # BGR for OpenCV
            
            elif fill_info['type'] == 'linear_gradient':
                # Linear gradient fill
                gradient = self.gradient_gen.create_linear_gradient(
                    image.shape,
                    fill_info['color1'][::-1],  # BGR
                    fill_info['color2'][::-1],  # BGR
                    fill_info['angle']
                )
                result[mask == 255] = gradient[mask == 255]
            
            elif fill_info['type'] == 'radial_gradient':
                # Radial gradient fill
                gradient = self.gradient_gen.create_radial_gradient(
                    image.shape,
                    fill_info['color1'][::-1],  # BGR
                    fill_info['color2'][::-1],  # BGR
                    fill_info['center']
                )
                result[mask == 255] = gradient[mask == 255]
            
            # Clean up mask immediately
            del mask
        
        # Draw back the black lines from original
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        black_lines = gray < 127
        result[black_lines] = [0, 0, 0]  # Keep black lines
        
        return result


class DesignColorizationSystem:
    """Main system - takes your B&W design and colorizes it"""
    
    def __init__(self, min_region_area=50, gradient_probability=0.3, max_image_size=2000):
        self.color_engine = ColorTheoryEngine()
        self.segmenter = DesignSegmenter(min_area=min_region_area, max_dimension=max_image_size)
        self.filler = SmartColorFiller(gradient_probability=gradient_probability)
    
    def process(self, design_path, output_dir='./outputs', num_variations=6):
        """
        Main processing pipeline:
        1. Load your black & white design
        2. Segment into regions
        3. Generate color palettes
        4. Apply colors and gradients
        5. Save variations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"  AI Design Colorization System")
        print(f"{'='*60}\n")
        
        # Load and prepare image
        print("[1/5] Loading your design...")
        image, gray = self.segmenter.load_and_prepare(design_path)
        print(f"      Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Segment design
        print("[2/5] Segmenting design into regions...")
        contours, binary = self.segmenter.segment_design(gray)
        masks = self.segmenter.create_region_masks(image.shape, contours)
        print(f"      Found {len(contours)} regions to colorize")
        
        # Save segmentation visualization
        seg_viz = image.copy()
        cv2.drawContours(seg_viz, contours, -1, (0, 255, 0), 2)
        seg_path = os.path.join(output_dir, 'segmentation_preview.png')
        cv2.imwrite(seg_path, seg_viz)
        print(f"      Saved segmentation preview")
        
        # Generate palettes
        print(f"[3/5] Generating {num_variations} color palettes...")
        palettes = self.color_engine.generate_multiple_palettes(
            num_palettes=num_variations,
            colors_per_palette=8
        )
        
        # Apply each palette
        print(f"[4/5] Creating colorized variations...")
        output_paths = []
        
        for i, palette in enumerate(palettes):
            print(f"      Variation {i+1}/{num_variations}...", end=' ')
            
            # Assign colors/gradients
            region_fills = self.filler.assign_colors_to_regions(contours, palette)
            
            # Apply fills
            colorized = self.filler.apply_fills_to_image(image, contours, masks, region_fills)
            
            # Save result
            output_path = os.path.join(output_dir, f'colorized_{i+1:02d}.png')
            cv2.imwrite(output_path, colorized)
            output_paths.append(output_path)
            
            # Save palette swatch
            palette_img = self.create_palette_swatch(palette)
            palette_path = os.path.join(output_dir, f'palette_{i+1:02d}.png')
            cv2.imwrite(palette_path, palette_img)
            
            # Clean up memory
            del colorized, palette_img, region_fills
            gc.collect()
            
            print("✓")
        
        print(f"[5/5] All done!")
        print(f"\n{'='*60}")
        print(f"  Created {len(output_paths)} colorized variations")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
        return output_paths
    
    def create_palette_swatch(self, palette, swatch_size=80):
        """Create a visual swatch of the palette"""
        num_colors = len(palette)
        swatch = np.zeros((swatch_size, swatch_size * num_colors, 3), dtype=np.uint8)
        
        for i, color in enumerate(palette):
            swatch[:, i*swatch_size:(i+1)*swatch_size] = color[::-1]  # BGR
        
        return swatch


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  AI Design Colorization System - Ready!")
    print("="*70)
    print("\nUsage:")
    print("  system = DesignColorizationSystem()")
    print("  system.process('your_design.png', output_dir='./outputs', num_variations=6)")
    print("\nSettings:")
    print("  - min_region_area: Minimum size of regions to color (default: 50)")
    print("  - gradient_probability: Chance of gradient vs solid (default: 0.3)")
    print("\nExample:")
    print("  system = DesignColorizationSystem(min_region_area=100, gradient_probability=0.4)")
    print("  system.process('my_mandala.png', num_variations=10)")
    print("="*70 + "\n")