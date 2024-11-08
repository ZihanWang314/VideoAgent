#!/usr/bin/env python3
import os
import logging
from PIL import Image
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import decord
from decord import VideoReader
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('frame_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_cuda_support():
    """Check if CUDA is properly supported by both torch and decord"""
    torch_cuda = torch.cuda.is_available()
    try:
        # Try to create a GPU context
        decord.bridge.set_bridge('torch')
        ctx = decord.gpu(0)
        # vr = VideoReader('dummy', ctx=decord.gpu(0))
        decord_cuda = True
    except:
        decord_cuda = False
    
    return torch_cuda and decord_cuda

def extract_frames_from_video(video_path: str, output_dir: str, device: str = 'cpu', fps: int = 1) -> None:
    """
    Extract frames from a video file at specified FPS using Decord.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        device: Device to use for processing ('cpu' or 'cuda')
        fps: Frames per second to extract (default: 1)
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up device context
        if device == 'cuda':
            decord.bridge.set_bridge('torch')  # Use PyTorch as bridge
            ctx = decord.gpu(0)
        else:
            ctx = decord.cpu(0)
            
        # Load video with specified device context
        vr = VideoReader(video_path, ctx=ctx)
        duration = len(vr)
        
        # Calculate frame indices to extract
        video_fps = vr.get_avg_fps()
        frame_interval = int(video_fps / fps)
        frame_indices = list(range(0, duration, frame_interval))
        
        # Extract frames in batches to improve efficiency
        batch_size = 32
        for batch_start in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[batch_start:batch_start + batch_size]
            
            # Get batch of frames
            frames = vr.get_batch(batch_indices)
            # if frame type is torch.tensor, convert to numpy array
            if isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            else:
                frames = frames.asnumpy()
            
            # Save frames
            for idx, frame in enumerate(frames):
                frame_number = batch_start + idx
                frame_path = os.path.join(output_dir, f"frame_{frame_number+1:06d}.jpg")
                # Save as JPEG
                # make frame 3 times smaller for width and height
                frame = Image.fromarray(frame).resize((frame.shape[1]//3, frame.shape[0]//3))
                frame.save(frame_path, "JPEG", quality=20)
        
        logger.info(f"Successfully extracted {len(frame_indices)} frames from {video_path}")
        
    except Exception as e:
        logger.error(f"Error processing {video_path}: {str(e)}")
        if device == 'cuda':
            logger.info(f"Retrying {video_path} with CPU...")
            extract_frames_from_video(video_path, output_dir, device='cpu', fps=fps)
        else:
            raise

def process_video(args):
    """Helper function for parallel processing"""
    video_path, output_base_dir, device, fps = args
    video_id = Path(video_path).stem
    output_dir = os.path.join(output_base_dir, video_id)
    try:
        # 看一下output_dir里面的文件数量是不是约等于video_path对应video的秒数，可以用metadata查看工具，会更快
        cur_len = len(list(Path(output_dir).glob('*.jpg')))
        vr = VideoReader(video_path)
        video_len = len(vr) // vr.get_avg_fps()
        if cur_len == video_len:
            logger.info(f"Frames already extracted for {video_path}")
            return video_id
        else:
            extract_frames_from_video(video_path, output_dir, device, fps)
            return video_id
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        # clean partially saved files
        raise e

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos using Decord')
    parser.add_argument('--input_dir', type=str, default='~/LongVideoBench/val_videos_short',
                      help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, default='frames',
                      help='Directory to save extracted frames')
    parser.add_argument('--workers', type=int, default=4,
                      help='Number of worker threads')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for frame extraction')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                      help='Device to use for processing (cpu or cuda)')
    parser.add_argument('--fps', type=int, default=1,
                      help='Frames per second to extract')
    args = parser.parse_args()
    
    # Check CUDA support if GPU is requested
    if args.device == 'cuda':
        raise NotImplementedError("CUDA support is not yet implemented")
        # if not check_cuda_support():
        #     logger.warning("CUDA support not properly enabled in either torch or decord. Falling back to CPU.")
        #     args.device = 'cpu'
        # else:
        #     logger.info("CUDA support verified for both torch and decord")
    
    # Expand user directory (~)
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return
    
    logger.info(f"Found {len(video_files)} videos to process")
    logger.info(f"Using device: {args.device}")
    
    # Process videos in parallel
    tasks = [(str(video), output_dir, args.device, args.fps) for video in video_files]
    
    # Adjust number of workers based on device
    effective_workers = args.workers if args.device == 'cpu' else 1
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        list(tqdm(
            executor.map(process_video, tasks),
            total=len(tasks),
            desc="Processing videos"
        ))
    
    logger.info("Frame extraction completed")

if __name__ == "__main__":
    main()