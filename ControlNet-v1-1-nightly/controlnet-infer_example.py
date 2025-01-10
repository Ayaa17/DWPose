# from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
# from controlnet_aux import OpenposeDetector
from annotator.dwpose import DWposeDetector
# from diffusers.utils import load_image
import cv2
from annotator.util import resize_image, HWC3

openpose = DWposeDetector()

# image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png")

image = cv2.imread("images_pose.png")
image = HWC3(image)
# image = resize_image(image, 512)
image = openpose(image)

# resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)  # (H, W, 3)
image_tensor = torch.from_numpy(image.copy()).float() / 255.0  # (H, W, 3)
# Step 3: 擴展為批次張量
num_samples = 1  # 批量大小
batch_tensor = torch.stack([image_tensor for _ in range(num_samples)], dim=0)  # (num_samples, H, W, 3)
# Step 4: 重排列維度
final_tensor = batch_tensor.permute(0, 3, 1, 2).clone()  # (num_samples, 3, H, W)
print(final_tensor.shape)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

image = pipe("chef in the kitchen", final_tensor, num_inference_steps=20).images[0]

image.save('images/chef_pose_out.png')
