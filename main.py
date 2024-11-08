import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image


from utils_clip import frame_retrieval_seg_ego
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


def read_caption(captions, sample_idx):
    return {f"frame {idx}": captions[idx - 1] for idx in sample_idx}


def run_one_question(video_id, ann, logs, llm_cache):#, blip_model, blip_processor, clip_model, clip_processor):
    question = ann["question"]
    answers = [ann[f"option {i}"] for i in range(5) if f"option {i}" in ann]
    formatted_question = (
        f"Here is the question: {question}\n"
        + "Here are the choices: "
        + " ".join([f"{i}. {ans}" for i, ans in enumerate(answers)])
    )
    frames_file_names = os.listdir(os.path.join("frames", video_id))
    num_frames = len(frames_file_names)

    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").cuda()
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

    ### Step 1 ###
    sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
    sampled_caps = []
    for idx in sample_idx:
        image_path = os.path.join("frames", video_id, f"frame_{idx:06d}.jpg")
        caption = generate_caption(image_path, blip_processor, blip_model)
        sampled_caps.append(caption)

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
            frame_idx = frame_retrieval_seg_ego(parsed_candidate_descriptions["frame_descriptions"], video_id, sample_idx)
            sample_idx = sorted(list(set(sample_idx + frame_idx)))
            sampled_caps = read_caption(caps, sample_idx)

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
            frame_idx = frame_retrieval_seg_ego(parsed_candidate_descriptions["frame_descriptions"], video_id, sample_idx)
            sample_idx = sorted(list(set(sample_idx + frame_idx)))
            sampled_caps = read_caption(caps, sample_idx)

            answer_str = generate_final_answer(formatted_question, sampled_caps, num_frames)
            answer = parse_text_find_number(answer_str)
        
    answer = answer if answer != -1 else random.randint(0, 4)

    label = int(ann["truth"])
    logs[video_id] = {
        "answer": answer,
        "label": label,
        "corr": int(label == answer),
        "count_frame": len(sample_idx),
    }


def main():
    # if running full set, change subset to fullset
    input_ann_file = "lvb_val_videoagent.json"
    json_file_name = "lvb_val_result.json"
    cache_llm_path = ".cache/llm_cache.jsonl"
    llm_cache = {}
    if os.path.exists(cache_llm_path):
        with open(cache_llm_path, "r") as f:
            for line in f:
                key, value = json.loads(line)
                llm_cache[key.encode()] = value.encode()

    anns = json.load(open(input_ann_file, "r"))
    logs = {}

    tasks = [(anns[video_id]['google_drive_id'], anns[video_id], logs, llm_cache) for video_id in anns.keys()]
    for task in tasks:
        run_one_question(*task)
        print("Processed successfully")

    json.dump(logs, open(json_file_name, "w"))

if __name__ == "__main__":
    main()
