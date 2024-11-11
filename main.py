import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from tqdm import tqdm
from openai import OpenAI
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


from utils_clip import CLIPFrameRetriever
from utils_blip import generate_caption
from utils_general import parse_json, parse_text_find_confidence, parse_text_find_number
from utils_llmeval import generate_final_answer, generate_description_step, ask_gpt_caption_step, ask_gpt_caption, self_eval, get_llm_response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line %(lineno)d)"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_caption(v_id, sample_idx, blip_processor, blip_model, cache_dir=".cache/captions"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, f"{v_id}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            sample_caps = json.load(f)
        sampled_caps = {k: v for k, v in sample_caps.items() if int(k.split(" ")[-1]) in sample_idx}
    else:
        sampled_caps = {}
    
    for idx in sample_idx:
        if f"frame {idx}" in sampled_caps:
            continue
        image_path = os.path.join("frames", v_id, f"frame_{idx:06d}.jpg")
        caption = generate_caption(image_path, blip_processor, blip_model)
        sampled_caps[f"frame {idx}"] = caption

    with open(cache_path, "w") as f:
        json.dump(sampled_caps, f)

    return sampled_caps



def run_one_question(ann, logs, llm_cache, answer_dir, blip_model, blip_processor, clip_retriever):
    clip_embed_only = False
    if clip_embed_only:
        video_id = ann["video_id"]
        # Set video-specific cache directory
        video_cache_path = os.path.join(clip_retriever.cache_dir, f"{video_id}.pt")

        if os.path.exists(video_cache_path):
            # Load cached tensor for the whole video
            video_frame_embeddings = torch.load(video_cache_path)
        else:
            frames_dir = "frames"
            if not os.path.exists(os.path.join(frames_dir, video_id)):
                return
            # Extract frame features for all frames in the video
            all_frame_paths = [
                os.path.join(frames_dir, video_id, i)
                for i in os.listdir(os.path.join(frames_dir, video_id))
            ]

            print(f"Extracting features for {video_id}")
            
            # Generate embeddings for all frames in one go
            video_frame_embeddings = clip_retriever.extract_frame_features(all_frame_paths).to(clip_retriever.device)
            
            # Save the entire video's frame embeddings as one tensor in .pt format
            torch.save(video_frame_embeddings, video_cache_path)
        return

    v_id = ann["video_id"]
    # if answer in logs, skip
    if ann['question_id'] in logs:
        print(f"Already processed {v_id}. load cache")
        return
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5) if f"option {i}" in ann]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    frames_file_names = os.listdir(os.path.join("frames", v_id))
    num_frames = len(frames_file_names)

    ### Step 1 ###
    sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
    all_sample_indexes = [sample_idx]

        
    sampled_caps = load_caption(v_id, sample_idx, blip_processor, blip_model)

    previous_prompt, answer_str = ask_gpt_caption(formatted_question, sampled_caps, num_frames, cache=llm_cache)
    answer = parse_text_find_number(answer_str)
    confidence_str = self_eval(previous_prompt, answer_str, cache=llm_cache)
    confidence = parse_text_find_confidence(confidence_str)


    ### Step 2 ###

    if confidence < 3:
        # try:
            segment_des = {i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}" for i in range(len(sample_idx) - 1)}
            candidate_descriptions = generate_description_step(formatted_question, sampled_caps, num_frames, segment_des, cache=llm_cache)
            parsed_candidate_descriptions = parse_json(candidate_descriptions)
            frame_idx = clip_retriever.frame_retrieval_seg_ego(parsed_candidate_descriptions["frame_descriptions"], v_id, sample_idx)
            all_sample_indexes.append(frame_idx)
            sample_idx = sorted(list(set(sample_idx + frame_idx)))
            sampled_caps = load_caption(v_id, sample_idx, blip_processor, blip_model)

            previous_prompt, answer_str = ask_gpt_caption_step(formatted_question, sampled_caps, num_frames, cache=llm_cache)
            answer = parse_text_find_number(answer_str)
            confidence_str = self_eval(previous_prompt, answer_str, cache=llm_cache)
            confidence = parse_text_find_confidence(confidence_str)

    ### Step 3 ###
    if confidence < 3:
        # try:
            segment_des = {i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}" for i in range(len(sample_idx) - 1)}
            candidate_descriptions = generate_description_step(formatted_question, sampled_caps, num_frames, segment_des, cache=llm_cache)
            parsed_candidate_descriptions = parse_json(candidate_descriptions)
            frame_idx = clip_retriever.frame_retrieval_seg_ego(parsed_candidate_descriptions["frame_descriptions"], v_id, sample_idx)
            all_sample_indexes.append(frame_idx)
            sample_idx = sorted(list(set(sample_idx + frame_idx)))
            sampled_caps = load_caption(v_id, sample_idx, blip_processor, blip_model)

            answer_str = generate_final_answer(formatted_question, sampled_caps, num_frames)
            answer = parse_text_find_number(answer_str)
        
    answer = answer if answer != -1 else random.randint(0, 4)

    label = int(ann["truth"])
    cur_log = {
        "question_id": ann['question_id'],
        "answer": answer,
        "label": label,
        "corr": int(label == answer),
        "count_frame": len(sample_idx),
        "confidence": confidence,
        "question": question,
        "frames": all_sample_indexes,
    }
    logs[ann['question_id']] = cur_log
    with open(answer_dir, "a") as f:
        json.dump(cur_log, f)
        f.write("\n")
    print(f"Processed {ann['question_id']} successfully")

def main():
    # if running full set, change subset to fullset
    input_ann_file = "lvb_val_videoagent.json"
    jsonl_file_name = "lvb_val_result.jsonl"
    cache_llm_path = ".cache/llm_cache.jsonl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_cache = {}
    if os.path.exists(cache_llm_path):
        with open(cache_llm_path, "r") as f:
            for line in f:
                key, value = json.loads(line)
                llm_cache[key.encode()] = value.encode()
    # open jsonl and load done videos

    logs = {}
    if os.path.exists(jsonl_file_name):
        with open(jsonl_file_name, "r") as f:
            for line in f:
                data = json.loads(line)
                logs[data["question_id"]] = data
    print("have already run", len(logs), "pieces of data")

    anns = json.load(open(input_ann_file, "r"))
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device).eval()
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    clip_retriever = CLIPFrameRetriever()

    tasks = [(anns[q_id], logs, llm_cache, jsonl_file_name, blip_model, blip_processor, clip_retriever) for q_id in anns]

    for task in tqdm(tasks):
        run_one_question(*task)
        print("Processed successfully")


if __name__ == "__main__":
    main()
