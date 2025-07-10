import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from ultralytics import YOLO
import torch
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import logging
import os
from dataclasses import dataclass
import json

@dataclass
class FaceFeatures:
    """Data class to store extracted face features"""
    landmarks: np.ndarray
    face_shape: str
    eye_shape: str
    skin_tone: Tuple[int, int, int]
    hair_color: Tuple[int, int, int]
    face_bbox: Tuple[int, int, int, int]
    confidence: float

@dataclass
class ImageAnalysis:
    """Data class to store complete image analysis results"""
    face_features: Optional[FaceFeatures]
    body_bbox: Optional[Tuple[int, int, int, int]]
    clothing_colors: List[Tuple[int, int, int]]
    accessories: List[str]
    missing_parts: List[str]
    suggestions: List[str]
    segmentation_masks: Dict[str, np.ndarray]
    image_size: Tuple[int, int]

class ImageAnalyzer:
    """
    Advanced image analyzer for character feature extraction
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._init_body_detection()
        self._init_mediapipe()
        
    def _init_body_detection(self):
        """Initialize body detection model"""
        try:
            self.body_model = YOLO('yolov8n.pt')
            self.logger.info("YOLOv8 initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLOv8: {e}")
            self.body_model = None
    
    def _init_mediapipe(self):
        """Initialize MediaPipe for facial landmarks"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.logger.info("MediaPipe FaceMesh initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe: {e}")
            self.face_mesh = None
    
    def analyze_image(self, image_path: str) -> ImageAnalysis:
        """
        Main method to analyze the input image
        """
        self.logger.info(f"Starting analysis of image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Analyze different aspects
        face_features = self._analyze_face(image_rgb)
        body_bbox = self._detect_body(image_rgb)
        clothing_colors = self._extract_clothing_colors(image_rgb, body_bbox)
        accessories = self._detect_accessories(image_rgb)
        missing_parts = self._check_missing_parts(image_rgb, face_features, body_bbox)
        suggestions = self._generate_suggestions(missing_parts, face_features)
        segmentation_masks = self._create_segmentation_masks(image_rgb, face_features, body_bbox)
        
        analysis = ImageAnalysis(
            face_features=face_features,
            body_bbox=body_bbox,
            clothing_colors=clothing_colors,
            accessories=accessories,
            missing_parts=missing_parts,
            suggestions=suggestions,
            segmentation_masks=segmentation_masks,
            image_size=(image.shape[1], image.shape[0])
        )
        
        self.logger.info("Image analysis completed successfully")
        return analysis
    
    def _analyze_face(self, image: np.ndarray) -> Optional[FaceFeatures]:
        """Analyze face features using MediaPipe"""
        if self.face_mesh is not None:
            return self._analyze_face_mediapipe(image)
        else:
            self.logger.warning("No face detection model available")
            return None
    
    def _analyze_face_mediapipe(self, image: np.ndarray) -> Optional[FaceFeatures]:
        """Analyze face using MediaPipe"""
        try:
            results = self.face_mesh.process(image)
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            landmarks_array = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] 
                                      for lm in landmarks.landmark])
            
            # Estimate bbox from landmarks
            bbox = self._landmarks_to_bbox(landmarks_array)
            
            # Extract features
            skin_tone = self._extract_skin_tone(image, bbox)
            hair_color = self._extract_hair_color(image, bbox)
            face_shape = self._classify_face_shape(landmarks_array)
            eye_shape = self._classify_eye_shape(landmarks_array)
            
            return FaceFeatures(
                landmarks=landmarks_array,
                face_shape=face_shape,
                eye_shape=eye_shape,
                skin_tone=skin_tone,
                hair_color=hair_color,
                face_bbox=bbox,
                confidence=0.8  # Default confidence for MediaPipe
            )
        except Exception as e:
            self.logger.error(f"Error in MediaPipe analysis: {e}")
            return None
    
    def _detect_body(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect body using YOLOv8"""
        if self.body_model is None:
            return None
        
        try:
            results = self.body_model(image, classes=[0])  # person class
            if len(results) == 0 or len(results[0].boxes) == 0:
                return None
            
            # Get the largest person detection
            boxes = results[0].boxes
            if len(boxes) == 0:
                return None
                
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            largest_idx = areas.argmax()
            bbox = boxes.xyxy[largest_idx].cpu().numpy().astype(int)
            
            return tuple(bbox)
        except Exception as e:
            self.logger.error(f"Error in body detection: {e}")
            return None
    
    def _extract_skin_tone(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """Extract dominant skin tone from face region"""
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return (128, 128, 128)  # Default gray
        
        # Convert to LAB color space for better skin tone analysis
        face_lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
        
        # Use k-means to find dominant color
        pixels = face_lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        
        # Get the most common cluster (likely skin)
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster = unique[counts.argmax()]
        
        skin_color_lab = kmeans.cluster_centers_[dominant_cluster]
        skin_color_rgb = cv2.cvtColor(np.uint8([[skin_color_lab]]), cv2.COLOR_LAB2RGB)[0][0]
        
        return tuple(skin_color_rgb)
    
    def _extract_hair_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """Extract hair color from upper face region"""
        x1, y1, x2, y2 = bbox
        # Focus on upper part of face for hair
        hair_region = image[y1:y1 + (y2-y1)//3, x1:x2]
        
        if hair_region.size == 0:
            return (64, 64, 64)  # Default dark gray
        
        # Convert to LAB and find dominant color
        hair_lab = cv2.cvtColor(hair_region, cv2.COLOR_RGB2LAB)
        pixels = hair_lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster = unique[counts.argmax()]
        
        hair_color_lab = kmeans.cluster_centers_[dominant_cluster]
        hair_color_rgb = cv2.cvtColor(np.uint8([[hair_color_lab]]), cv2.COLOR_LAB2RGB)[0][0]
        
        return tuple(hair_color_rgb)
    
    def _classify_face_shape(self, landmarks: np.ndarray) -> str:
        """Classify face shape based on landmarks"""
        # Simple classification based on face width/height ratio
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        ratio = face_width / face_height
        
        if ratio > 0.85:
            return "round"
        elif ratio < 0.7:
            return "long"
        else:
            return "oval"
    
    def _classify_eye_shape(self, landmarks: np.ndarray) -> str:
        """Classify eye shape based on landmarks"""
        # This is a simplified classification
        # In a real implementation, you'd use more sophisticated analysis
        return "almond"  # Default classification
    
    def _extract_clothing_colors(self, image: np.ndarray, body_bbox: Optional[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int]]:
        """Extract clothing colors from body region"""
        if body_bbox is None:
            return [(128, 128, 128)]  # Default gray
        
        x1, y1, x2, y2 = body_bbox
        body_region = image[y1:y2, x1:x2]
        
        if body_region.size == 0:
            return [(128, 128, 128)]
        
        # Convert to LAB and find dominant colors
        body_lab = cv2.cvtColor(body_region, cv2.COLOR_RGB2LAB)
        pixels = body_lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.config['analyzer']['feature_extraction']['color_clusters'], random_state=42)
        kmeans.fit(pixels)
        
        colors = []
        for center in kmeans.cluster_centers_:
            color_rgb = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_LAB2RGB)[0][0]
            colors.append(tuple(color_rgb))
        
        return colors
    
    def _detect_accessories(self, image: np.ndarray) -> List[str]:
        """Detect accessories like glasses, hats, etc."""
        accessories = []
        
        # Simple detection based on color and shape analysis
        # In a real implementation, you'd use object detection models
        # Convert to uint8 to avoid OpenCV depth issues
        image_uint8 = image.astype(np.uint8)
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        
        # Detect dark regions that might be glasses
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Reasonable size for glasses
                accessories.append("glasses")
                break
        
        return accessories
    
    def _check_missing_parts(self, image: np.ndarray, face_features: Optional[FaceFeatures], body_bbox: Optional[Tuple[int, int, int, int]]) -> List[str]:
        """Check for missing body parts"""
        missing_parts = []
        
        if face_features is None:
            missing_parts.append("face")
            return missing_parts
        
        # Check if top of head is visible
        x1, y1, x2, y2 = face_features.face_bbox
        if y1 < 10:  # Too close to top edge
            missing_parts.append("top_of_head")
        
        # Check if shoulders are visible
        if body_bbox is None:
            missing_parts.append("shoulders")
        else:
            bx1, by1, bx2, by2 = body_bbox
            if by2 - y2 < 50:  # Not enough body below face
                missing_parts.append("shoulders")
        
        return missing_parts
    
    def _generate_suggestions(self, missing_parts: List[str], face_features: Optional[FaceFeatures]) -> List[str]:
        """Generate suggestions based on missing parts and analysis"""
        suggestions = []
        
        for part in missing_parts:
            if part == "top_of_head":
                suggestions.append("Consider inpainting to add hair/head top")
            elif part == "shoulders":
                suggestions.append("Consider inpainting to add shoulders and upper body")
            elif part == "face":
                suggestions.append("Face not detected - check image quality and lighting")
        
        if face_features:
            if face_features.face_shape == "long":
                suggestions.append("Consider variations with different face angles")
            # Check for glasses without calling _detect_accessories with zeros
            suggestions.append("Consider adding glasses as a variation")
        
        return suggestions
    
    def _create_segmentation_masks(self, image: np.ndarray, face_features: Optional[FaceFeatures], body_bbox: Optional[Tuple[int, int, int, int]]) -> Dict[str, np.ndarray]:
        """Create segmentation masks for different regions"""
        height, width = image.shape[:2]
        masks = {}
        
        # Create face mask
        if face_features:
            face_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = face_features.face_bbox
            face_mask[y1:y2, x1:x2] = 255
            masks['face'] = face_mask
        
        # Create body mask
        if body_bbox:
            body_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = body_bbox
            body_mask[y1:y2, x1:x2] = 255
            masks['body'] = body_mask
        
        # Create background mask (inverse of body)
        if body_bbox:
            background_mask = np.ones((height, width), dtype=np.uint8) * 255
            x1, y1, x2, y2 = body_bbox
            background_mask[y1:y2, x1:x2] = 0
            masks['background'] = background_mask
        
        return masks
    
    def _landmarks_to_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert landmarks to bounding box"""
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def save_analysis(self, analysis: ImageAnalysis, output_path: str):
        """Save analysis results to JSON file"""
        analysis_dict = {
            "face_features": {
                "face_bbox": analysis.face_features.face_bbox if analysis.face_features else None,
                "face_shape": analysis.face_features.face_shape if analysis.face_features else None,
                "landmarks": analysis.face_features.landmarks.tolist() if analysis.face_features and analysis.face_features.landmarks is not None else None
            },
            "body_bbox": analysis.body_bbox,
            "color_analysis": {
                "dominant_colors": analysis.clothing_colors,
                "color_palette": analysis.clothing_colors
            },
            "accessories": analysis.accessories,
            "segmentation_masks": {
                key: mask.tolist() if mask is not None else None 
                for key, mask in analysis.segmentation_masks.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis_dict, f, indent=2)
    
    def load_analysis(self, input_path: str) -> ImageAnalysis:
        """Load analysis results from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct face features
        face_features = None
        if data.get("face_features") and data["face_features"].get("face_bbox"):
            face_features = FaceFeatures(
                face_bbox=data["face_features"]["face_bbox"],
                face_shape=data["face_features"]["face_shape"],
                landmarks=np.array(data["face_features"]["landmarks"]) if data["face_features"]["landmarks"] else None
            )
        
        # Reconstruct color analysis
        color_analysis = None
        if data.get("color_analysis"):
            color_analysis = ColorAnalysis(
                dominant_colors=data["color_analysis"]["dominant_colors"],
                color_palette=np.array(data["color_analysis"]["color_palette"]) if data["color_analysis"]["color_palette"] else None
            )
        
        # Reconstruct segmentation masks
        segmentation_masks = {}
        for key, mask_data in data.get("segmentation_masks", {}).items():
            if mask_data is not None:
                segmentation_masks[key] = np.array(mask_data)
        
        return ImageAnalysis(
            face_features=face_features,
            body_bbox=data.get("body_bbox"),
            clothing_colors=data.get("color_analysis", {}).get("dominant_colors", []),
            accessories=data.get("accessories", []),
            segmentation_masks=segmentation_masks
        ) 