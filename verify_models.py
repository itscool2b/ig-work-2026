import torch
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.io import decode_image
from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

img = decode_image('image.jpg')

# cuda
print(f"CUDA: {torch.cuda.get_device_name(0)}")

# resnet50
weights_rn = ResNet50_Weights.IMAGENET1K_V2
model_rn = resnet50(weights=weights_rn)
model_rn.eval()
batch_rn = weights_rn.transforms()(img).unsqueeze(0)
pred = model_rn(batch_rn).squeeze(0).softmax(0)
print(f"ResNet50: {weights_rn.meta['categories'][pred.argmax().item()]}")
del model_rn

# vit-b/16
weights_vit = ViT_B_16_Weights.IMAGENET1K_V1
model_vit = vit_b_16(weights=weights_vit)
model_vit.eval()
batch_vit = weights_vit.transforms()(img).unsqueeze(0)
pred = model_vit(batch_vit).squeeze(0).softmax(0)
print(f"ViT-B/16: {weights_vit.meta['categories'][pred.argmax().item()]}")
del model_vit

# tinyllama
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model_tl = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")
tokens = tokenizer("Hello", return_tensors="pt").to("cuda")
out = model_tl.generate(**tokens, max_new_tokens=10)
print(f"TinyLlama: {tokenizer.decode(out[0], skip_special_tokens=True)}")
del model_tl
torch.cuda.empty_cache()

# llava 4-bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_lv = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", quantization_config=bnb_config, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
print(f"LLaVA: loaded ({sum(p.numel() for p in model_lv.parameters()) / 1e9:.1f}B params, 4-bit)")
del model_lv
torch.cuda.empty_cache()

print("all models verified")
