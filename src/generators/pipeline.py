#!/usr/bin/env python3
"""
Main Pipeline Orchestrator for Synthetic Dataset Generation
"""

import os
import logging
import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from .image_analyzer import ImageAnalyzer, ImageAnalysis
from .latent_encoder import LatentEncoder
from .prompt_generator import PromptGenerator

@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline"""
    input_image: str
    output_dir: str
    temp_dir: str
    target_dataset_size: int
    max_generated_images: int
    save_intermediates: bool = True

class SyntheticDatasetPipeline:
    """
    Main pipeline for generating synthetic dataset from a single image
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.pipeline_config = self._create_pipeline_config()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.image_analyzer = ImageAnalyzer(self.config)
        
        self.logger.info("Pipeline initialized successfully")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_pipeline_config(self) -> PipelineConfig:
        """Create pipeline configuration from loaded config"""
        return PipelineConfig(
            input_image=self.config['pipeline']['input_image'],
            output_dir=self.config['pipeline']['output_dir'],
            temp_dir=self.config['pipeline']['temp_dir'],
            target_dataset_size=self.config['pipeline']['target_dataset_size'],
            max_generated_images=self.config['pipeline']['max_generated_images'],
            save_intermediates=self.config['logging']['save_intermediates']
        )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config['logging']['log_level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.pipeline_config.output_dir, 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
    
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.pipeline_config.output_dir, exist_ok=True)
        os.makedirs(self.pipeline_config.temp_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['analysis', 'latents', 'generated', 'filtered', 'final']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.pipeline_config.output_dir, subdir), exist_ok=True)
    
    def run_step_1_image_analysis(self) -> ImageAnalysis:
        """
        Step 1: Analyze the input image
        """
        self.logger.info("="*50)
        self.logger.info("STEP 1: IMAGE ANALYSIS")
        self.logger.info("="*50)
        
        # Analyze image
        analysis = self.image_analyzer.analyze_image(self.pipeline_config.input_image)
        
        # Save analysis results
        analysis_path = os.path.join(self.pipeline_config.output_dir, 'analysis', 'analysis_results.json')
        self.image_analyzer.save_analysis(analysis, analysis_path)
        
        # Save visualization
        self._save_analysis_visualization(analysis)
        
        self.logger.info("Step 1 completed successfully")
        return analysis
    
    def _save_analysis_visualization(self, analysis: ImageAnalysis):
        """Save analysis visualization"""
        try:
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np
            
            # Load original image
            image = cv2.imread(self.pipeline_config.input_image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Image Analysis Results', fontsize=16)
            
            # Original image
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Face detection
            if analysis.face_features:
                face_img = image_rgb.copy()
                x1, y1, x2, y2 = analysis.face_features.face_bbox
                cv2.rectangle(face_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                axes[0, 1].imshow(face_img)
                axes[0, 1].set_title(f'Face Detection\nShape: {analysis.face_features.face_shape}')
                axes[0, 1].axis('off')
            else:
                axes[0, 1].imshow(image_rgb)
                axes[0, 1].set_title('No Face Detected')
                axes[0, 1].axis('off')
            
            # Body detection
            if analysis.body_bbox:
                body_img = image_rgb.copy()
                x1, y1, x2, y2 = analysis.body_bbox
                cv2.rectangle(body_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                axes[0, 2].imshow(body_img)
                axes[0, 2].set_title('Body Detection')
                axes[0, 2].axis('off')
            else:
                axes[0, 2].imshow(image_rgb)
                axes[0, 2].set_title('No Body Detected')
                axes[0, 2].axis('off')
            
            # Segmentation masks
            if 'face' in analysis.segmentation_masks:
                axes[1, 0].imshow(analysis.segmentation_masks['face'], cmap='gray')
                axes[1, 0].set_title('Face Mask')
                axes[1, 0].axis('off')
            
            if 'body' in analysis.segmentation_masks:
                axes[1, 1].imshow(analysis.segmentation_masks['body'], cmap='gray')
                axes[1, 1].set_title('Body Mask')
                axes[1, 1].axis('off')
            
            if 'background' in analysis.segmentation_masks:
                axes[1, 2].imshow(analysis.segmentation_masks['background'], cmap='gray')
                axes[1, 2].set_title('Background Mask')
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            viz_path = os.path.join(self.pipeline_config.output_dir, 'analysis', 'analysis_visualization.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Analysis visualization saved to {viz_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save analysis visualization: {e}")
    
    def run_step_2_latent_encoding(self, analysis: ImageAnalysis):
        """
        Step 2: Latent Space Encoding & Interpolation
        """
        self.logger.info("="*50)
        self.logger.info("STEP 2: LATENT SPACE ENCODING & INTERPOLATION")
        self.logger.info("="*50)
        
        from PIL import Image
        import torch
        import os
        import numpy as np
        
        # 1. Load input image
        image_path = self.pipeline_config.input_image
        image = Image.open(image_path)
        
        # 2. Initialize LatentEncoder
        latent_encoder = LatentEncoder()
        
        # 3. Encode image to latent
        self.logger.info("Encoding input image to latent...")
        latents = latent_encoder.image_to_latent(image)
        
        # 4. Generate variations using fast method
        self.logger.info("Generating latent variations with noise...")
        n_variations = 6  # Зменшено з 8 для прискорення
        latent_variations = latent_encoder.generate_variations_fast(latents, n_variations)
        
        # 5. Save latents
        os.makedirs(os.path.join(self.pipeline_config.output_dir, 'latents'), exist_ok=True)
        base_latent_path = os.path.join(self.pipeline_config.output_dir, 'latents', 'input_latent.pt')
        torch.save(latents, base_latent_path)
        for i, lvar in enumerate(latent_variations):
            torch.save(lvar, os.path.join(self.pipeline_config.output_dir, 'latents', f'latent_noise_{i}.pt'))
        self.logger.info(f"Saved {len(latent_variations)+1} latents to output/latents/")
        
        # 6. Save only a few preview images for speed
        try:
            # Save only input latent and first variation as preview
            preview_latents = [latents, latent_variations[0]]
            for i, lvar in enumerate(preview_latents):
                img = latent_encoder.latent_to_image(lvar)
                img.save(os.path.join(self.pipeline_config.output_dir, 'latents', f'latent_preview_{i}.png'))
            self.logger.info("Saved 2 latent preview images.")
        except Exception as e:
            self.logger.warning(f"Could not save latent preview images: {e}")
        
        self.logger.info("Step 2 completed successfully")
        return {
            'input_latent': base_latent_path,
            'latent_variations': [os.path.join(self.pipeline_config.output_dir, 'latents', f'latent_noise_{i}.pt') for i in range(n_variations)]
        }
    
    def run_step_3_prompt_injection(self):
        """
        Step 3: Dynamic Prompt Injection
        """
        self.logger.info("="*50)
        self.logger.info("STEP 3: DYNAMIC PROMPT INJECTION")
        self.logger.info("="*50)
        
        import os
        import json
        
        # Load analysis results from step 1
        analysis_path = os.path.join(self.pipeline_config.output_dir, 'analysis', 'analysis_results.json')
        if not os.path.exists(analysis_path):
            self.logger.error("Analysis results not found. Please run step 1 first.")
            return None
        
        # Load analysis
        analysis = self.image_analyzer.load_analysis(analysis_path)
        
        # Initialize prompt generator
        prompt_generator = PromptGenerator()
        
        # Generate prompts
        self.logger.info("Generating dynamic prompts based on image analysis...")
        num_prompts = 15  # Зменшено з 25 для прискорення
        prompts = prompt_generator.generate_prompts(analysis, num_prompts)
        
        # Save prompts
        prompts_dir = os.path.join(self.pipeline_config.output_dir, 'prompts')
        os.makedirs(prompts_dir, exist_ok=True)
        
        prompts_path = os.path.join(prompts_dir, 'generated_prompts.json')
        prompt_generator.save_prompts(prompts, prompts_path)
        
        # Save prompt statistics
        stats = {
            "total_prompts": len(prompts),
            "poses_used": list(set(p["components"]["pose"] for p in prompts)),
            "expressions_used": list(set(p["components"]["expression"] for p in prompts)),
            "clothing_used": list(set(p["components"]["clothing"] for p in prompts)),
            "backgrounds_used": list(set(p["components"]["background"] for p in prompts)),
            "styles_used": list(set(p["components"]["style"] for p in prompts)),
            "face_shape": analysis.face_features.face_shape if analysis.face_features else "unknown",
            "dominant_colors": analysis.color_analysis.dominant_colors if analysis.color_analysis else []
        }
        
        stats_path = os.path.join(prompts_dir, 'prompt_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Log some example prompts
        self.logger.info(f"Generated {len(prompts)} prompts")
        self.logger.info("Example prompts:")
        for i, prompt in enumerate(prompts[:3]):
            self.logger.info(f"  {i+1}. {prompt['full_prompt']}")
        
        self.logger.info(f"Prompts saved to {prompts_path}")
        self.logger.info(f"Statistics saved to {stats_path}")
        self.logger.info("Step 3 completed successfully")
        
        return {
            'prompts_file': prompts_path,
            'statistics_file': stats_path,
            'prompts': prompts
        }
    
    def run_step_4_inpainting_controlnet(self):
        """
        Step 4: Selective Inpainting + ControlNet
        """
        self.logger.info("="*50)
        self.logger.info("STEP 4: SELECTIVE INPAINTING + CONTROLNET")
        self.logger.info("="*50)
        
        # TODO: Implement inpainting and ControlNet
        self.logger.info("Step 4: Inpainting and ControlNet - TO BE IMPLEMENTED")
        
        return None
    
    def run_step_5_filtering(self):
        """
        Step 5: Augmentation-aware Filtering
        """
        self.logger.info("="*50)
        self.logger.info("STEP 5: AUGMENTATION-AWARE FILTERING")
        self.logger.info("="*50)
        
        # TODO: Implement filtering
        self.logger.info("Step 5: Filtering - TO BE IMPLEMENTED")
        
        return None
    
    def run_full_pipeline(self):
        """
        Run the complete pipeline
        """
        self.logger.info("Starting full synthetic dataset generation pipeline")
        
        try:
            # Step 1: Image Analysis
            analysis = self.run_step_1_image_analysis()
            
            # Step 2: Latent Encoding & Interpolation
            latents = self.run_step_2_latent_encoding(analysis)
            
            # Step 3: Prompt Injection
            prompts = self.run_step_3_prompt_injection()
            
            # Step 4: Inpainting & ControlNet
            inpainted = self.run_step_4_inpainting_controlnet()
            
            # Step 5: Filtering
            filtered = self.run_step_5_filtering()
            
            self.logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function to run the pipeline"""
    pipeline = SyntheticDatasetPipeline()
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main() 