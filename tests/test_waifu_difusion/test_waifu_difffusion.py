import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    torch_dtype=torch.float32
).to('cuda')

def dummy_checker(images, **kwargs): return images, False
pipe.safety_checker = dummy_checker

prompt = "1girl, naked, aqua eyes, sexy, pussy"
with autocast("cuda"):
    images = pipe(prompt, guidance_scale=6).images
    


print(f"total {len(images)} images.")

for i in range(len(images)):
    image = images[i]
    image.save("test_{}.png".format(i))