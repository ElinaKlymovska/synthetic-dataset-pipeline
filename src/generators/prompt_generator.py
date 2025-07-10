#!/usr/bin/env python3
"""
Dynamic Prompt Generator for Synthetic Dataset Generation
"""

import random
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from .image_analyzer import ImageAnalysis

@dataclass
class PromptTemplate:
    """Template for generating prompts"""
    base_prompt: str
    pose_variations: List[str]
    expression_variations: List[str]
    clothing_variations: List[str]
    background_variations: List[str]
    style_variations: List[str]

class PromptGenerator:
    """
    Generates dynamic prompts based on image analysis
    """
    
    def __init__(self, clip_model: str = "all-MiniLM-L6-v2"):
        self.clip_model = SentenceTransformer(clip_model)
        self.templates = self._load_templates()
        
    def _load_templates(self) -> PromptTemplate:
        """Load prompt templates"""
        return PromptTemplate(
            base_prompt="portrait of a person, high quality, detailed",
            pose_variations=[
                "looking straight ahead", "looking to the side", "looking up", "looking down",
                "head slightly tilted", "three-quarter view", "profile view", "over-the-shoulder look",
                "candid pose", "formal pose", "casual pose", "dynamic pose"
            ],
            expression_variations=[
                "neutral expression", "smiling", "serious expression", "confident expression",
                "thoughtful expression", "happy expression", "calm expression", "focused expression",
                "friendly expression", "professional expression", "casual expression", "natural expression"
            ],
            clothing_variations=[
                "casual clothing", "formal clothing", "business attire", "casual wear",
                "smart casual", "elegant clothing", "simple clothing", "stylish clothing",
                "professional outfit", "everyday clothing", "trendy clothing", "classic clothing"
            ],
            background_variations=[
                "blurred background", "studio background", "outdoor background", "indoor background",
                "natural background", "urban background", "simple background", "professional background",
                "clean background", "minimal background", "abstract background", "neutral background"
            ],
            style_variations=[
                "photorealistic", "cinematic lighting", "studio lighting", "natural lighting",
                "professional photography", "high resolution", "detailed", "sharp focus",
                "beautiful lighting", "artistic", "modern style", "classic style"
            ]
        )
    
    def generate_prompts(self, analysis: ImageAnalysis, num_prompts: int = 15) -> List[Dict]:
        """
        Generate diverse prompts based on image analysis
        """
        prompts = []
        
        # Extract features from analysis
        face_shape = analysis.face_features.face_shape if analysis.face_features else "oval"
        dominant_colors = analysis.clothing_colors if hasattr(analysis, 'clothing_colors') else ["neutral"]
        
        for i in range(num_prompts):
            prompt_data = self._generate_single_prompt(analysis, face_shape, dominant_colors, i)
            prompts.append(prompt_data)
        
        return prompts
    
    def _generate_single_prompt(self, analysis: ImageAnalysis, face_shape: str, 
                               dominant_colors: List[str], index: int) -> Dict:
        """Generate a single prompt with variations"""
        
        # Base components
        pose = random.choice(self.templates.pose_variations)
        expression = random.choice(self.templates.expression_variations)
        clothing = random.choice(self.templates.clothing_variations)
        background = random.choice(self.templates.background_variations)
        style = random.choice(self.templates.style_variations)
        
        # Add face shape specific details
        face_detail = f"{face_shape} face shape" if face_shape != "unknown" else ""
        
        # Add color information
        color_detail = f"wearing {dominant_colors[0]} clothing" if dominant_colors else ""
        
        # Combine into full prompt
        full_prompt = f"{self.templates.base_prompt}, {pose}, {expression}, {clothing}, {background}, {style}"
        if face_detail:
            full_prompt += f", {face_detail}"
        if color_detail:
            full_prompt += f", {color_detail}"
        
        # Generate CLIP embedding only for first few prompts to save time
        embedding = None
        if index < 5:  # Only generate embeddings for first 5 prompts
            embedding = self._get_clip_embedding(full_prompt)
        
        return {
            "id": f"prompt_{index:03d}",
            "full_prompt": full_prompt,
            "components": {
                "pose": pose,
                "expression": expression,
                "clothing": clothing,
                "background": background,
                "style": style,
                "face_shape": face_shape,
                "colors": dominant_colors
            },
            "clip_embedding": embedding.tolist() if embedding is not None else None,
            "metadata": {
                "generated_from_analysis": True,
                "face_detected": analysis.face_features is not None,
                "body_detected": analysis.body_bbox is not None
            }
        }
    
    def _get_clip_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get CLIP embedding for text"""
        try:
            embedding = self.clip_model.encode(text)
            return embedding
        except Exception as e:
            print(f"Warning: Could not generate CLIP embedding: {e}")
            return None
    
    def save_prompts(self, prompts: List[Dict], output_path: str):
        """Save generated prompts to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)
    
    def load_prompts(self, input_path: str) -> List[Dict]:
        """Load prompts from JSON file"""
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def get_prompt_embeddings(self, prompts: List[Dict]) -> np.ndarray:
        """Extract CLIP embeddings from prompts"""
        embeddings = []
        for prompt in prompts:
            if prompt.get("clip_embedding"):
                embeddings.append(prompt["clip_embedding"])
        return np.array(embeddings) if embeddings else np.array([]) 