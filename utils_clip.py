from PIL import Image
import numpy as np
from typing import List, Union
import os

class CLIPFrameRetriever:
    def __init__(self, model_name: str = "ViT-g-14", pretrained: str = "laion2b_s12b_b42k"):
        """
        Initialize CLIP model for online feature generation.
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained model weights to use
        """
        import torch
        import open_clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
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
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # Extract features
                image_features = self.model.encode_image(image_input)
                image_features = image_features.cpu().numpy()
                
                # Normalize features
                image_features = image_features / np.linalg.norm(image_features)
                features.append(image_features.squeeze())
                
        return np.array(features)
    
    def extract_text_features(self, descriptions: List[str]) -> np.ndarray:
        """
        Extract CLIP features from text descriptions.
        
        Args:
            descriptions: List of text descriptions
            
        Returns:
            Array of normalized CLIP features for texts
        """
        with torch.no_grad():
            text = self.tokenizer(descriptions).to(self.device)
            text_features = self.model.encode_text(text)
            text_features = text_features.cpu().numpy()
            
            # Normalize features
            text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
            
        return text_features

def frame_retrieval_seg_ego(descriptions: List[dict], video_id: str, sample_idx: List[int], 
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
    # Initialize CLIP retriever
    retriever = CLIPFrameRetriever()
    
    # Extract descriptions
    description_list = []
    for desc in descriptions:
        for key, value in desc.items():
            if "description" in key:
                description_list.append(value)
                break
    
    # Get text embeddings
    text_embedding = retriever.extract_text_features(description_list)
    
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg = int(description["segment_id"]) - 1
        
        # Get frame paths for current segment
        seg_frame_paths = [
            os.path.join(frames_dir, video_id, f"frame_{i:06d}.jpg")
            for i in range(sample_idx[seg], sample_idx[seg + 1])
        ]
        
        if len(seg_frame_paths) < 2:
            frame_idx.append(sample_idx[seg] + 1)
            continue
            
        # Extract frame features for segment
        seg_frame_embeddings = retriever.extract_frame_features(seg_frame_paths)
        
        # Calculate similarities and get best matching frame
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)
    
    return frame_idx