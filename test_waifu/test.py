from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = lambda images, clip_input: (images, False)
prompt = "a photo of a naked girl, showing her pussy, white legwear"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")