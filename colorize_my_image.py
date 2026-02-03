"""
Colorize YOUR Black & White Design
Just run: python colorize_my_image.py
"""

from design_colorizer import DesignColorizationSystem

# ============================================
# CHANGE THESE SETTINGS TO MATCH YOUR NEEDS
# ============================================

# 1. PUT YOUR IMAGE PATH HERE (e.g., 'my_design.png' or 'C:/Users/YourName/Desktop/mandala.png')
YOUR_IMAGE_PATH = '737-SCAN-NEW.tif'  # ← CHANGE THIS!

# 2. Where to save colorized outputs
OUTPUT_FOLDER = './my_colorized_outputs'

# 3. How many different colorized versions to create
NUM_VARIATIONS = 6

# 4. Settings
MIN_REGION_SIZE = 150       # Larger = ignores small details (try 50-200)
GRADIENT_CHANCE = 0.4       # 0.0 = only solid colors, 1.0 = only gradients
MAX_IMAGE_SIZE = 2000       # Maximum width/height (larger images auto-resize)

# ============================================
# RUN THE COLORIZATION
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  🎨 AI Design Colorization System")
    print("="*70 + "\n")
    
    # Initialize the system
    print(f"📂 Input file: {YOUR_IMAGE_PATH}")
    print(f"💾 Output folder: {OUTPUT_FOLDER}")
    print(f"🎨 Creating {NUM_VARIATIONS} variations")
    print(f"⚙️  Gradient probability: {GRADIENT_CHANCE*100}%")
    print(f"📏 Max image size: {MAX_IMAGE_SIZE}px (auto-resize if larger)\n")
    
    system = DesignColorizationSystem(
        min_region_area=MIN_REGION_SIZE,
        gradient_probability=GRADIENT_CHANCE,
        max_image_size=MAX_IMAGE_SIZE
    )
    
    # Process your image
    try:
        output_paths = system.process(
            design_path=YOUR_IMAGE_PATH,
            output_dir=OUTPUT_FOLDER,
            num_variations=NUM_VARIATIONS
        )
        
        print("\n✅ SUCCESS!")
        print(f"   Created {len(output_paths)} colorized versions")
        print(f"   Check the '{OUTPUT_FOLDER}' folder\n")
        print("="*70 + "\n")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find file '{YOUR_IMAGE_PATH}'")
        print("   Please check:")
        print("   1. File path is correct")
        print("   2. File exists")
        print("   3. Use full path if needed (e.g., 'C:/Users/Name/image.png')\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("   Check that your image is a valid black & white design\n")