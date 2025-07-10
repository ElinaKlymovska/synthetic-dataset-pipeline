#!/usr/bin/env python3
"""
Run Character Generation Script
Generates character variations for LoRA training using the new refactored architecture
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from app import GenImgApp, create_app_with_dependencies
from interfaces import GenerationConfig

def main():
    """Main function to run character generation"""
    
    # Check for source image
    source_image = "data/input/character_reference.jpg"
    
    if not os.path.exists(source_image):
        print(f"Source image not found: {source_image}")
        print("Please place your source image in the data/input/ directory")
        return 1
    
    print("=== Character Generation Pipeline ===")
    print(f"Source image: {source_image}")
    
    try:
        # Create app with dependencies
        print("Initializing generator...")
        app = create_app_with_dependencies()
        
        # Create generation config
        config = GenerationConfig(
            prompt="A professional headshot of the character in different poses and lighting",
            num_images=15,
            output_dir="data/output",
            save_metadata=True
        )
        
        # Generate variations
        print("\nGenerating character variations...")
        with app:
            results = app.generate_character_variations(source_image, config)
        
        if results.generated_images:
            print(f"\n✅ Successfully generated {len(results.generated_images)} images")
            print("Generated images:")
            for path in results.generated_images:
                print(f"  - {path}")
            
            print(f"💰 Total cost: ${results.total_cost:.2f}")
            print(f"⏱️ Generation time: {results.generation_time}")
            
            print("\n=== Generation Complete ===")
            print("You can now use the generated images for LoRA training")
            print(f"Metadata saved to: {config.output_dir}")
            
        else:
            print("❌ No images were generated successfully")
            return 1
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 