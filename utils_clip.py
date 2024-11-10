from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from typing import List, Union
import os
import torch

class CLIPFrameRetriever:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", cache_dir: str = ".cache/clip_cache"):
        """
        Initialize CLIP model for online feature generation.
        
        Args:
            model_name: CLIP model architecture from transformers
            cache_dir: Directory to cache frame features
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def extract_frame_features(self, frame_paths: List[str]) -> np.ndarray:
        """
        Extract CLIP features from video frames.
        
        Args:
            frame_paths: List of paths to frame images
            
        Returns:
            Array of normalized CLIP features for frames
        """
        features = []
        with torch.no_grad():
            for frame_path in frame_paths:
                # Load and preprocess image
                image = Image.open(frame_path).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                # Extract features
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features.cpu()
                
                # Normalize features
                image_features = image_features / torch.linalg.norm(image_features)
                features.append(image_features.squeeze())
                
        return torch.stack(features)
    
    def extract_text_features(self, descriptions: List[str]) -> np.ndarray:
        """
        Extract CLIP features from text descriptions.
        
        Args:
            descriptions: List of text descriptions
            
        Returns:
            Array of normalized CLIP features for texts
        """
        with torch.no_grad():
            inputs = self.processor(text=descriptions, return_tensors="pt", padding=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / torch.linalg.norm(text_features, axis=1, keepdims=True)
            
        return text_features

    def frame_retrieval_seg_ego(self, descriptions: List[dict], video_id: str, sample_idx: List[int], 
                                frames_dir: str = "frames") -> List[int]:
        """
        Retrieve relevant frames based on segment descriptions using online CLIP feature generation.
        
        Args:
            descriptions: List of segment descriptions
            video_id: ID of the video being processed
            sample_idx: List of sample frame indices
            frames_dir: Directory containing frame images
            
        Returns:
            List of retrieved frame indices
        """  
        # Set video-specific cache directory
        video_cache_path = os.path.join(self.cache_dir, f"{video_id}.pt")
        
        # Extract descriptions
        description_list = []
        for desc in descriptions:
            for key, value in desc.items():
                if "description" in key:
                    description_list.append(value)
                    break
        
        # Get text embeddings
        text_embedding = self.extract_text_features(description_list)
        
        if os.path.exists(video_cache_path):
            # Load cached tensor for the whole video
            video_frame_embeddings = torch.load(video_cache_path)
        else:
            # Extract frame features for all frames in the video
            all_frame_paths = [
                os.path.join(frames_dir, video_id, f"frame_{i:06d}.jpg")
                for i in sample_idx
            ]
            
            # Generate embeddings for all frames in one go
            video_frame_embeddings = self.extract_frame_features(all_frame_paths).to(self.device)
            
            # Save the entire video's frame embeddings as one tensor in .pt format
            torch.save(video_frame_embeddings, video_cache_path)

        frame_idx = []
        for idx, description in enumerate(descriptions):
            # Calculate similarities for the entire video and get the best matching frame index
            seg_similarity = text_embedding[idx] @ video_frame_embeddings.T
            best_frame_index = seg_similarity.argmax().item()
            frame_idx.append(sample_idx[best_frame_index])


        return frame_idx
