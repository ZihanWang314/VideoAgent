from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import json
from torch.profiler import profile, record_function, ProfilerActivity

# def generate_caption(image_path, blip_processor, blip_model):
#     image = Image.open(image_path).convert("RGB")
    
#     with torch.profiler.profile(
#         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#         profile_memory=True,
#         with_flops=True
#     ) as profiler: # 这个能分析性能，但是太慢了，大概能知道处理一个frame的FLOPS. 之后average一下放进paper去。
#         inputs = blip_processor(images=image, return_tensors="pt")
#         caption_ids = blip_model.generate(**{k:v.to(blip_model.device) for k, v in inputs.items()})
#         caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    
#     stats = profiler.key_averages()

#     # Print FLOPs per operation
#     stats_filtered = [stat for stat in stats if stat.flops is not None and stat.flops != 0]
    
#     with open("features_profile.jsonl", "a") as f:
#         f.write(json.dumps(extract_important_stats(stats_filtered)) + "\n")

#     return caption

def generate_caption(image_path, blip_processor, blip_model):
    image = Image.open(image_path).convert("RGB")
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     profile_memory=True,
    #     with_flops=True
    # ) as profiler: # 这个能分析性能，但是太慢了，大概能知道处理一个frame的FLOPS. 之后average一下放进paper去。
    #     with record_function("model_inference"):
    inputs = blip_processor(images=image, return_tensors="pt")
    caption_ids = blip_model.generate(**{k:v.to(blip_model.device) for k, v in inputs.items()}, do_sample=False, max_new_tokens=128)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    # stats = profiler.key_averages()
    # cpu_time, cuda_time = stats.table().split("\n")[-3:-1]
    # cpu_time, cuda_time = cpu_time.split(" ")[-1], cuda_time.split(" ")[-1]
    # stats_filtered = [stat for stat in stats if stat.flops is not None and stat.flops != 0]
    # flops = sum([stat.flops for stat in stats_filtered])
    # with open("blip_caption_features_profile.jsonl", "a") as f:
        # f.write(json.dumps({"cpu_time": cpu_time, "cuda_time": cuda_time, "flops": flops}) + "\n")

    return caption

if __name__ == "__main__":
    blip_name = "Salesforce/blip-image-captioning-large"
    device="cuda" if torch.cuda.is_available() else "cpu"
    blip_processor = BlipProcessor.from_pretrained(blip_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_name, torch_dtype=torch.float16).to(device).eval()
    image_path = "frames/Yh48p5efBTk/frame_000003.jpg"
    caption = generate_caption(image_path, blip_processor, blip_model)
    print("生成的描述:", caption)