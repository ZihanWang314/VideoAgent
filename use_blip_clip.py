from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# clip_name = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
clip_name = "openai/clip-vit-base-patch32"
blip_name = "Salesforce/blip-image-captioning-large"
# clip is for image retrieval, blip is for captioning

blip_processor = BlipProcessor.from_pretrained(blip_name)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_name)

clip_processor = CLIPProcessor.from_pretrained(clip_name)
clip_model = CLIPModel.from_pretrained(clip_name)

# 加载图像
image = Image.open("frames/Yh48p5efBTk/frame_000003.jpg").convert("RGB")

# 处理图像并生成描述
inputs = processor(images=image, return_tensors="pt")
caption_ids = model.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)

print("生成的描述:", caption)

# 使用 CLIP 验证描述和图像的一致性
inputs = clip_processor(text=caption, images=image, return_tensors="pt", padding=True)
outputs = clip_model(**inputs)
logits_per_image = outputs.logits_per_image
similarity = logits_per_image.item()

print("描述与图像的相似度:", similarity)