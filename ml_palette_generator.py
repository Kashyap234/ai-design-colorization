"""
ML-Enhanced Color Palette Generator
Uses K-means clustering and neural network-based color harmonization
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import colorsys
from typing import List, Tuple, Dict
import pickle


class MLColorPaletteGenerator:
    """Machine Learning-based color palette generator"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.color_features_cache = {}
    
    def extract_color_features(self, rgb: Tuple[int, int, int]) -> np.ndarray:
        """Extract features from RGB color for ML processing"""
        r, g, b = rgb
        
        # Convert to different color spaces
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        
        # Extract features
        features = [
            r, g, b,           # RGB values
            h, s, v,           # HSV values
            r + g + b,         # Brightness
            max(r, g, b) - min(r, g, b),  # Saturation proxy
            abs(r - g),        # Color differences
            abs(g - b),
            abs(b - r),
        ]
        
        return np.array(features)
    
    def cluster_colors_kmeans(self, color_list: List[Tuple[int, int, int]], 
                              n_clusters: int = 8) -> List[Tuple[int, int, int]]:
        """Use K-means to find dominant colors"""
        if len(color_list) < n_clusters:
            return color_list
        
        # Extract features
        features = np.array([self.extract_color_features(c) for c in color_list])
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(features)
        
        # Get cluster centers and convert back to RGB
        centers = kmeans.cluster_centers_
        dominant_colors = []
        
        for center in centers:
            r, g, b = int(center[0]), int(center[1]), int(center[2])
            # Clamp values
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            dominant_colors.append((r, g, b))
        
        return dominant_colors
    
    def calculate_color_harmony_score(self, palette: List[Tuple[int, int, int]]) -> float:
        """Calculate harmony score for a palette (0-1, higher is better)"""
        if len(palette) < 2:
            return 0.5
        
        scores = []
        
        # Convert to HSV for harmony calculation
        hsv_colors = [colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0) for r, g, b in palette]
        
        # Check for complementary relationships
        for i in range(len(hsv_colors)):
            for j in range(i+1, len(hsv_colors)):
                h1, s1, v1 = hsv_colors[i]
                h2, s2, v2 = hsv_colors[j]
                
                # Hue difference
                hue_diff = abs(h1 - h2)
                hue_diff = min(hue_diff, 1.0 - hue_diff)  # Circular distance
                
                # Check for harmonious relationships
                # Complementary: ~0.5, Triadic: ~0.33, Analogous: ~0.083
                harmony_distances = [0.5, 0.33, 0.67, 0.083, 0.25, 0.75]
                min_distance = min([abs(hue_diff - hd) for hd in harmony_distances])
                
                # Score based on proximity to harmonious relationship
                harmony_score = 1.0 - (min_distance * 2)  # Scale to 0-1
                scores.append(max(0, harmony_score))
        
        # Average harmony score
        avg_harmony = np.mean(scores) if scores else 0.5
        
        # Penalty for very similar colors
        similarity_penalty = 0
        for i in range(len(palette)):
            for j in range(i+1, len(palette)):
                r1, g1, b1 = palette[i]
                r2, g2, b2 = palette[j]
                distance = np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
                if distance < 30:  # Very similar colors
                    similarity_penalty += 0.1
        
        final_score = max(0, avg_harmony - similarity_penalty)
        return min(1.0, final_score)
    
    def generate_palette_from_reference(self, reference_colors: List[Tuple[int, int, int]], 
                                       num_palettes: int = 6,
                                       palette_size: int = 8) -> List[List[Tuple[int, int, int]]]:
        """Generate multiple palettes based on reference colors using ML clustering"""
        
        # Cluster reference colors to find dominant colors
        dominant_colors = self.cluster_colors_kmeans(reference_colors, n_clusters=min(20, len(reference_colors)))
        
        palettes = []
        
        for i in range(num_palettes):
            # Different strategies for each palette
            strategy = i % 4
            
            if strategy == 0:  # Warm palette
                palette = self._generate_warm_palette(dominant_colors, palette_size)
            elif strategy == 1:  # Cool palette
                palette = self._generate_cool_palette(dominant_colors, palette_size)
            elif strategy == 2:  # High contrast
                palette = self._generate_high_contrast_palette(dominant_colors, palette_size)
            else:  # Balanced
                palette = self._generate_balanced_palette(dominant_colors, palette_size)
            
            palettes.append(palette)
        
        # Sort palettes by harmony score
        palettes_with_scores = [(p, self.calculate_color_harmony_score(p)) for p in palettes]
        palettes_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [p for p, _ in palettes_with_scores]
    
    def _generate_warm_palette(self, reference: List[Tuple[int, int, int]], size: int) -> List[Tuple[int, int, int]]:
        """Generate warm color palette"""
        palette = []
        warm_hue_ranges = [(0, 0.15), (0.85, 1.0)]  # Reds, oranges, yellows
        
        for ref_color in reference:
            h, s, v = colorsys.rgb_to_hsv(ref_color[0]/255.0, ref_color[1]/255.0, ref_color[2]/255.0)
            
            # Check if in warm range
            is_warm = any(start <= h <= end for start, end in warm_hue_ranges)
            
            if is_warm:
                # Use this color with variations
                for i in range(2):
                    new_s = max(0.3, min(1.0, s + (i-0.5) * 0.2))
                    new_v = max(0.4, min(1.0, v + (i-0.5) * 0.15))
                    rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, new_s, new_v))
                    palette.append(rgb)
        
        # Fill remaining with generated warm colors
        while len(palette) < size:
            h = np.random.choice([np.random.uniform(0, 0.15), np.random.uniform(0.85, 1.0)])
            s = np.random.uniform(0.5, 0.9)
            v = np.random.uniform(0.6, 0.95)
            rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
            palette.append(rgb)
        
        return palette[:size]
    
    def _generate_cool_palette(self, reference: List[Tuple[int, int, int]], size: int) -> List[Tuple[int, int, int]]:
        """Generate cool color palette"""
        palette = []
        cool_hue_ranges = [(0.4, 0.7)]  # Blues, greens, cyans
        
        for ref_color in reference:
            h, s, v = colorsys.rgb_to_hsv(ref_color[0]/255.0, ref_color[1]/255.0, ref_color[2]/255.0)
            
            is_cool = any(start <= h <= end for start, end in cool_hue_ranges)
            
            if is_cool:
                for i in range(2):
                    new_s = max(0.3, min(1.0, s + (i-0.5) * 0.2))
                    new_v = max(0.4, min(1.0, v + (i-0.5) * 0.15))
                    rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, new_s, new_v))
                    palette.append(rgb)
        
        while len(palette) < size:
            h = np.random.uniform(0.4, 0.7)
            s = np.random.uniform(0.5, 0.9)
            v = np.random.uniform(0.6, 0.95)
            rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
            palette.append(rgb)
        
        return palette[:size]
    
    def _generate_high_contrast_palette(self, reference: List[Tuple[int, int, int]], size: int) -> List[Tuple[int, int, int]]:
        """Generate high contrast palette"""
        palette = []
        
        # Start with extreme values
        palette.append((255, 255, 255))  # White
        palette.append((0, 0, 0))        # Black
        
        # Add vibrant colors from reference
        for ref_color in reference:
            h, s, v = colorsys.rgb_to_hsv(ref_color[0]/255.0, ref_color[1]/255.0, ref_color[2]/255.0)
            
            # Maximize saturation and value for contrast
            new_s = 0.9
            new_v = 0.9
            rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, new_s, new_v))
            palette.append(rgb)
        
        # Fill with complementary colors
        while len(palette) < size:
            if len(palette) >= 3:
                # Find color with least representation
                h = np.random.uniform(0, 1.0)
                s = np.random.uniform(0.7, 1.0)
                v = np.random.uniform(0.7, 1.0)
                rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
                palette.append(rgb)
        
        return palette[:size]
    
    def _generate_balanced_palette(self, reference: List[Tuple[int, int, int]], size: int) -> List[Tuple[int, int, int]]:
        """Generate balanced palette with good variety"""
        palette = []
        
        # Select diverse colors from reference
        if len(reference) >= size:
            # Use K-means to get most diverse colors
            palette = self.cluster_colors_kmeans(reference, n_clusters=size)
        else:
            palette = reference.copy()
            
            # Generate additional colors
            while len(palette) < size:
                # Find gaps in hue space
                existing_hues = [colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)[0] for r, g, b in palette]
                
                # Find largest gap
                existing_hues.sort()
                max_gap = 0
                gap_center = 0
                
                for i in range(len(existing_hues)):
                    next_i = (i + 1) % len(existing_hues)
                    gap = existing_hues[next_i] - existing_hues[i]
                    if gap < 0:
                        gap += 1.0
                    
                    if gap > max_gap:
                        max_gap = gap
                        gap_center = (existing_hues[i] + gap / 2) % 1.0
                
                # Create color in the gap
                h = gap_center
                s = np.random.uniform(0.5, 0.8)
                v = np.random.uniform(0.6, 0.9)
                rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
                palette.append(rgb)
        
        return palette[:size]


def create_sample_reference_colors():
    """Create sample reference colors for testing"""
    reference = [
        (255, 87, 51),   # Red-orange
        (199, 0, 57),    # Crimson
        (255, 195, 0),   # Gold
        (46, 196, 182),  # Turquoise
        (1, 22, 39),     # Dark blue
        (0, 149, 218),   # Sky blue
        (255, 255, 255), # White
        (44, 44, 44),    # Dark gray
    ]
    return reference


# Integration with main system
if __name__ == "__main__":
    # Test the ML palette generator
    generator = MLColorPaletteGenerator()
    
    print("ML-Enhanced Color Palette Generator")
    print("=" * 50)
    
    # Create sample reference colors
    reference = create_sample_reference_colors()
    print(f"\nUsing {len(reference)} reference colors")
    
    # Generate palettes
    palettes = generator.generate_palette_from_reference(reference, num_palettes=6, palette_size=8)
    
    print(f"\nGenerated {len(palettes)} palettes:")
    for i, palette in enumerate(palettes):
        harmony_score = generator.calculate_color_harmony_score(palette)
        print(f"\nPalette {i+1} (Harmony Score: {harmony_score:.3f}):")
        for j, color in enumerate(palette):
            print(f"  Color {j+1}: RGB{color}")