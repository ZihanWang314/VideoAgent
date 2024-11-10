from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


def generate_caption(image_path, blip_processor, blip_model):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    caption_ids = blip_model.generate(**{k:v.to(blip_model.device) for k, v in inputs.items()})
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    blip_name = "Salesforce/blip-image-captioning-large"
    device="cuda" if torch.cuda.is_available() else "cpu"
    blip_processor = BlipProcessor.from_pretrained(blip_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_name, torch_dtype=torch.float16).to(device).eval()
    image_path = "frames/Yh48p5efBTk/frame_000003.jpg"
    caption = generate_caption(image_path, blip_processor, blip_model)
    print("生成的描述:", caption)