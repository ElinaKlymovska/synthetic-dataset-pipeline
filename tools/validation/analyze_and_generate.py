#!/usr/bin/env python3
"""
Character Analysis and Identity-Preserving Generation Script
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from identity_preserving_generator import IdentityPreservingGenerator

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('character_generation.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Analyze character and generate identity-preserving variations")
    parser.add_argument("--reference", required=True, help="Path to reference character image")
    parser.add_argument("--count", type=int, default=15, help="Number of variations to generate")
    parser.add_argument("--output", default="data/output", help="Output directory for generated images")
    parser.add_argument("--analysis-output", default="data/analysis", help="Output directory for analysis")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze character, don't generate")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Check if reference image exists
    if not os.path.exists(args.reference):
        logger.error(f"Reference image not found: {args.reference}")
        return 1
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.analysis_output, exist_ok=True)
    
    try:
        # Initialize generator
        generator = IdentityPreservingGenerator()
        
        # Analyze character
        logger.info("🔍 Starting character analysis...")
        analysis, character_prompt = generator.analyze_and_store_identity(
            args.reference, args.analysis_output
        )
        
        # Print analysis summary
        print("\n" + "="*60)
        print("📊 CHARACTER ANALYSIS SUMMARY")
        print("="*60)
        print(f"Image: {args.reference}")
        print(f"Size: {analysis.get('image_size', 'Unknown')}")
        
        # Facial features
        facial = analysis.get("facial_features", {})
        if facial:
            print("\n👤 FACIAL FEATURES:")
            if facial.get("face_shape"):
                print(f"  Face shape: {facial['face_shape']}")
            if facial.get("eye_features"):
                eyes = facial["eye_features"]
                print(f"  Eyes: {eyes.get('eye_color', 'Unknown')} {eyes.get('eye_shape', 'Unknown')}")
            if facial.get("mouth_features"):
                mouth = facial["mouth_features"]
                print(f"  Lips: {mouth.get('lip_shape', 'Unknown')}, Smile: {mouth.get('smile_type', 'Unknown')}")
            if facial.get("hair_features"):
                hair = facial["hair_features"]
                print(f"  Hair: {hair.get('hair_color', 'Unknown')} {hair.get('hair_style', 'Unknown')}")
        
        # Body features
        body = analysis.get("body_features", {})
        if body:
            print("\n🏃 BODY FEATURES:")
            if body.get("body_type"):
                print(f"  Body type: {body['body_type']}")
            if body.get("chest_measurements"):
                chest = body["chest_measurements"]
                print(f"  Chest: {chest.get('chest_size', 'Unknown')} size, {chest.get('chest_shape', 'Unknown')} shape")
        
        # Skin analysis
        skin = analysis.get("skin_analysis", {})
        if skin:
            print("\n🎨 SKIN ANALYSIS:")
            if skin.get("skin_tone_category"):
                print(f"  Skin tone: {skin['skin_tone_category']}")
            if skin.get("skin_undertone"):
                print(f"  Undertone: {skin['skin_undertone']}")
        
        # Missing features
        missing = analysis.get("missing_features", [])
        if missing:
            print("\n⚠️  MISSING FEATURES:")
            for feature in missing:
                print(f"  - {feature}")
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  - {rec}")
        
        # Identity strength
        if generator.character_embedding:
            strength = generator.character_embedding.get("identity_strength", 0)
            print(f"\n💪 IDENTITY STRENGTH: {strength:.2f}/1.0")
        
        print(f"\n🎭 CHARACTER PROMPT: {character_prompt}")
        print("="*60)
        
        if args.analyze_only:
            logger.info("✅ Analysis complete. Use --generate to create variations.")
            return 0
        
        # Generate variations
        logger.info(f"🎨 Generating {args.count} identity-preserving variations...")
        
        # Define pose and outfit variations
        pose_variations = [
            "full body front view, standing pose, looking at camera, natural expression",
            "full body back view, standing pose, elegant posture, looking over shoulder",
            "full body side view, standing pose, profile shot, elegant",
            "full body front view, blowing kiss, playful expression, hand gesture",
            "full body front view, lying on bed, front view, relaxed pose",
            "full body side view, lying on bed, side view, elegant curve",
            "full body back view, lying on bed, back view, graceful pose",
            "full body front view, sitting pose, elegant, legs crossed",
            "full body front view, walking pose, dynamic, confident stride",
            "full body front view, dancing pose, graceful, flowing movement",
            "full body front view, casual standing, natural, relaxed",
            "full body front view, fashion pose, confident, model stance",
            "full body front view, portrait pose, close up, intimate",
            "full body front view, full length shot, elegant, sophisticated",
            "full body front view, three quarter view, dynamic, engaging"
        ]
        
        outfit_variations = [
            "elegant dress, formal wear, sophisticated, high quality fabric",
            "bikini, swimwear, beach style, summer fashion",
            "lingerie, intimate wear, elegant, silk fabric",
            "business suit, professional attire, formal, tailored fit",
            "Hinata from Naruto cosplay, orange outfit, ninja style, anime character",
            "cocktail dress, evening wear, glamorous, party attire",
            "red long dress with cleavage, elegant evening gown, formal event",
            "casual dress, everyday wear, comfortable, relaxed style",
            "party dress, festive attire, celebration, fun fashion",
            "summer dress, light fabric, breezy, warm weather style",
            "winter dress, warm fabric, cozy, cold weather fashion",
            "formal gown, red carpet style, luxurious, high fashion",
            "minimalist dress, simple design, clean, modern style",
            "vintage dress, retro style, classic, timeless fashion",
            "modern dress, contemporary fashion, trendy, current style"
        ]
        
        generated_images = generator.generate_identity_preserving_variations(
            args.reference,
            args.output,
            args.count,
            pose_variations,
            outfit_variations
        )
        
        # Summary
        print("\n" + "="*60)
        print("🎉 GENERATION SUMMARY")
        print("="*60)
        print(f"✅ Successfully generated: {len(generated_images)}/{args.count} images")
        print(f"📁 Output directory: {args.output}")
        print(f"📊 Analysis saved to: {args.analysis_output}")
        
        if generated_images:
            print("\n📸 Generated images:")
            for i, img_path in enumerate(generated_images, 1):
                print(f"  {i:2d}. {os.path.basename(img_path)}")
        
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during generation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 