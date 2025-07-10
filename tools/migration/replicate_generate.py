#!/usr/bin/env python3
"""
Replicate API Character Generation Script
Generate diverse character images using the new refactored CLI system.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cli import CLIManager

def main():
    """Main function for character generation using the new CLI system."""
    parser = argparse.ArgumentParser(
        description="Generate character images using Replicate API"
    )
    
    parser.add_argument(
        "--reference", 
        default="data/input/character_reference.jpg",
        help="Path to reference character image"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="data/output",
        help="Output directory for generated images"
    )
    
    parser.add_argument(
        "--count", 
        type=int, 
        default=15,
        help="Number of images to generate"
    )
    
    parser.add_argument(
        "--config", 
        default="config/replicate_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--estimate-cost", 
        action="store_true",
        help="Estimate cost without generating"
    )
    
    parser.add_argument(
        "--check-api", 
        action="store_true",
        help="Check API connection and token"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Check if API token is set
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN environment variable not set!")
        print("💡 Get your token from: https://replicate.com/account/api-tokens")
        print("💡 Set it with: export REPLICATE_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Create CLI manager
    cli = CLIManager(config_path=args.config)
    
    try:
        # Handle different modes
        if args.check_api:
            print("🔍 Checking environment...")
            cli.env_check()
            return
        
        if args.estimate_cost:
            print("💰 Estimating generation cost...")
            cli.estimate(args.count)
            return
        
        # Validate reference image
        if not Path(args.reference).exists():
            print(f"❌ Reference image not found: {args.reference}")
            sys.exit(1)
        
        # Show generation info
        print("🎨 Character Generation Setup:")
        print(f"   Reference: {args.reference}")
        print(f"   Output: {args.output_dir}")
        print(f"   Count: {args.count}")
        print(f"   Config: {args.config}")
        
        # Generate images
        if args.verbose:
            print("🎯 Starting character generation...")
        
        # Use the CLI generate command
        cli.generate(
            reference_image=args.reference,
            count=args.count,
            output_dir=args.output_dir,
            prompt="Professional character variations for LoRA training"
        )
        
        print("✅ Generation completed successfully!")
        
    except KeyboardInterrupt:
        print("⏹️ Generation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 